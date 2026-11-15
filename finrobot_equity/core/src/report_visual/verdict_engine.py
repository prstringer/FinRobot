#!/usr/bin/env python
# coding: utf-8
"""
Verdict engine: turns (value, own history, peer median) into a verdict object using ``verdicts.yaml``.

Rules (from the brief):
* verdict is good | neutral | concern; direction (higher or lower is better) and every threshold live in the YAML.
* a verdict always has a yardstick and a scale; if no yardstick can be applied, ``evaluate`` returns None.
* general norms are the last resort and are labelled "General norm, not company-specific".
* ``so_what`` explains what the figure means; templates may use computed facts but never restate the metric's own value.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ICON = {"good": "check", "neutral": "dash", "concern": "alert"}
GENERAL_NORM_TEXT = "General norm, not company-specific"
YARDSTICK_TEXT = {"own_history": "vs own history", "peers": "vs peers", "general_norm": GENERAL_NORM_TEXT}


@lru_cache(maxsize=1)
def load_config(path: Optional[str] = None) -> dict:
    """verdicts.yaml, plus any ``verdicts_*.yaml`` variant files (e.g. verdicts_bank.yaml) merged into ``metrics``."""
    with open(path or os.path.join(HERE, "verdicts.yaml"), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is None:
        for name in sorted(os.listdir(HERE)):
            if name.startswith("verdicts_") and name.endswith(".yaml"):
                with open(os.path.join(HERE, name), "r", encoding="utf-8") as fh:
                    extra = yaml.safe_load(fh) or {}
                for k, v in (extra.get("metrics") or {}).items():
                    cfg["metrics"].setdefault(k, v)
                for k, v in (extra.get("technical_tests") or {}).items():
                    cfg.setdefault("technical_tests", {}).setdefault(k, v)
    return cfg


@lru_cache(maxsize=1)
def load_glossary(path: Optional[str] = None) -> list[dict]:
    with open(path or os.path.join(HERE, "glossary.yaml"), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fmt_value(unit: str, v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    if unit == "%":
        return f"{v:.1f}%"
    if unit == "x":
        return f"{v:.2f}x" if abs(v) < 5 else f"{v:.1f}x"
    if unit == "Rs":
        return f"Rs {v:,.0f}"
    if unit == "Rs per Rs 100":
        return f"Rs {v:.0f} per Rs 100"
    return f"{v:,.1f} {unit}".strip()


class _Safe(dict):
    def __missing__(self, key):
        raise KeyError(key)


def so_what_text(cfg: dict, verdict: str, facts: Optional[dict]) -> str:
    tpl = cfg["so_what"][verdict]
    try:
        return tpl.format_map(_Safe(facts or {}))
    except (KeyError, ValueError):
        # a template that needs a fact we do not have falls back to the generic sentence for that verdict
        return {"good": "This is a strength against its yardstick.", "neutral": "This is in line with its yardstick.",
                "concern": "This is a weakness against its yardstick."}[verdict]


def _by_own_history(cfg, value, hist, higher):
    d = load_config()["defaults"]["own_history"]
    pts = [h for h in (hist or []) if h is not None]
    if len(pts) < d["min_points"]:
        return None
    lo, hi = min(pts), max(pts)
    if hi == lo:
        return "neutral", {"min": lo, "max": hi}
    pos = (value - lo) / (hi - lo)
    pos = pos if higher else 1 - pos
    v = "good" if pos >= d["good_pos"] else ("concern" if pos <= d["concern_pos"] else "neutral")
    return v, {"min": lo, "max": hi}


def _by_peers(cfg, value, peer_median, higher):
    if peer_median in (None, 0):
        return None
    p = {**load_config()["defaults"]["peer"], **(cfg.get("peer") or {})}
    rel = value / peer_median
    if higher:
        v = "good" if rel >= 1 + p["tol_good"] else ("concern" if rel <= 1 - p["tol_bad"] else "neutral")
    else:
        v = "good" if rel <= 1 - p["tol_good"] else ("concern" if rel >= 1 + p["tol_bad"] else "neutral")
    return v, {}


def _by_norm(cfg, value, higher):
    n = cfg.get("general_norm")
    if not n:
        return None
    if higher:
        v = "good" if value >= n["good"] else ("concern" if value < n["concern"] else "neutral")
    else:
        v = "good" if value <= n["good"] else ("concern" if value > n["concern"] else "neutral")
    return v, {}


def evaluate(key: str, value: Optional[float], history: Optional[list] = None, peer_median: Optional[float] = None,
             facts: Optional[dict] = None, period: Optional[str] = None, calc_extra: Optional[str] = None,
             scale_override: Optional[dict] = None, forced: Optional[dict] = None) -> Optional[dict]:
    """Verdict object for one metric, or None when no yardstick applies.

    ``forced`` lets a custom rule (operating leverage, cash conversion, fair value) supply verdict and yardstick.
    """
    cfg = load_config()["metrics"].get(key)
    if cfg is None or value is None:
        return None
    higher = bool(cfg["higher_is_better"])
    hist = [h for h in (history or []) if h is not None]
    verdict = yardstick = None
    scale = {"min": min(hist) if hist else None, "max": max(hist) if hist else None}
    if forced:
        verdict, yardstick = forced["verdict"], forced["yardstick"]
        scale.update(forced.get("scale") or {})
    else:
        for y in cfg["yardsticks"]:
            res = (_by_peers(cfg, value, peer_median, higher) if y == "peers" else
                   _by_own_history(cfg, value, hist, higher) if y == "own_history" else
                   _by_norm(cfg, value, higher) if y == "general_norm" else None)
            if res:
                verdict, yardstick, extra = res[0], y, res[1]
                scale.update({k: v for k, v in extra.items() if v is not None})
                break
    if verdict is None:
        return None
    if scale_override:
        scale.update(scale_override)
    scale.update({"peer_median": peer_median, "higher_is_better": higher, "value": value})
    if scale.get("min") is None or scale.get("max") is None:
        scale["min"], scale["max"] = (min(value, peer_median or value), max(value, peer_median or value))
    unit = cfg["unit"]
    source = "general_norm" if yardstick == "general_norm" else cfg.get("source", "report_data")
    return {
        "id": key, "metric": key, "label": cfg["label"], "value": value, "value_text": fmt_value(unit, value), "unit": unit,
        "period": period, "verdict": verdict, "icon": ICON[verdict], "yardstick": yardstick,
        "yardstick_text": YARDSTICK_TEXT[yardstick], "scale": scale, "so_what": so_what_text(cfg, verdict, facts),
        "source": source,
        "calc": cfg["calc"] + (f" {calc_extra}" if calc_extra else ""),
    }


# ----------------------------------------------------------------------------------------------
# custom rules
# ----------------------------------------------------------------------------------------------
def operating_leverage(per100: Optional[float], avg_ebitda_margin_pct: Optional[float], facts: dict, period: str) -> Optional[dict]:
    if per100 is None or not avg_ebitda_margin_pct:
        return None
    cfg = load_config()["metrics"]["operating_leverage"]
    t = cfg["thresholds"]
    ratio = abs(per100) / avg_ebitda_margin_pct
    verdict = "good" if ratio <= t["good_ratio"] else ("concern" if ratio >= t["concern_ratio"] else "neutral")
    scale = {"min": 0.0, "max": max(60.0, abs(per100) * 1.1)}
    f = dict(facts or {}, per100=abs(per100))
    return evaluate("operating_leverage", abs(per100), [], None, f, period,
                    calc_extra=f"Average EBITDA margin over the window is {avg_ebitda_margin_pct:.1f}%, the proportional move.",
                    forced={"verdict": verdict, "yardstick": "own_history", "scale": scale})


def cash_conversion(latest_pct: Optional[float], hist_pct: list, aggregate_pct: Optional[float], cyclical_flag: bool,
                    period: str) -> Optional[dict]:
    if latest_pct is None:
        return None
    cfg = load_config()["metrics"]["cash_conversion"]
    flatter = cfg["thresholds"]["flatter_ratio"]
    n = cfg["general_norm"]
    base = aggregate_pct if aggregate_pct is not None else latest_pct
    if cyclical_flag and aggregate_pct and latest_pct >= flatter * aggregate_pct:
        verdict, yardstick = "concern", "own_history"
    else:
        verdict = "good" if base >= n["good"] else ("concern" if base < n["concern"] else "neutral")
        yardstick = "general_norm"
    return evaluate("cash_conversion", latest_pct, hist_pct, None, {}, period,
                    calc_extra=(f"Five-year aggregate is {aggregate_pct:.0f}%." if aggregate_pct is not None else None),
                    forced={"verdict": verdict, "yardstick": yardstick})


def fair_value(price: Optional[float], bands: list[tuple[float, float]], period: str) -> Optional[dict]:
    if price is None or not bands:
        return None
    lo, hi = min(b[0] for b in bands), max(b[1] for b in bands)
    verdict = "concern" if price > hi else ("good" if price < lo else "neutral")
    return evaluate("fair_value", price, [], None, {}, period,
                    forced={"verdict": verdict, "yardstick": "peers", "scale": {"min": lo, "max": hi}})
