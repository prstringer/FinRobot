#!/usr/bin/env python
# coding: utf-8
"""
Bank metric providers: everything that turns the bank context and tables into verdict chips, fair-value cases, signposts
and traffic lights. What to compute is read from ``variants/bank.yaml``; thresholds from ``verdicts_bank.yaml``.
"""

from __future__ import annotations

import os
import statistics
from typing import Optional

import yaml

from .. import verdict_engine as ve

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANT_PATH = os.path.join(os.path.dirname(HERE), "variants", "bank.yaml")


def load_variant(path: Optional[str] = None) -> dict:
    with open(path or VARIANT_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fy(label: Optional[str]) -> Optional[str]:
    if label and len(label) == 5 and label[:4].isdigit():
        return f"FY{label[2:4]}{'E' if label[4] == 'E' else ''}"
    return label


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def actual_series(ctx: dict, name: str) -> dict[str, float]:
    """Actual-year values (oldest to newest) of one row of the bank analysis table."""
    s = (ctx.get("bank_series") or {}).get(name) or {}
    return {k: v for k, v in sorted(s.items()) if k.endswith("A")}


def quantile(xs: list[float], stat: str) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if stat == "min":
        return xs[0]
    if stat == "max":
        return xs[-1]
    if stat == "median":
        return float(statistics.median(xs))
    if len(xs) < 2:
        return xs[0]
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[0] if stat == "q1" else q[2]


# ----------------------------------------------------------------------------------------------
# facts
# ----------------------------------------------------------------------------------------------
def compute_facts(ctx: dict, variant: dict) -> dict:
    latest = ctx["meta"]["latest_actual_year"]
    f: dict = {"latest_year": latest, "latest_label": fy(latest), "years": sorted(actual_series(ctx, "EPS"))}
    cv, mk = ctx["current_valuation"], ctx["market"]
    f.update({"price": mk.get("price"), "shares_mn": mk.get("shares_mn"), "eps_ttm": cv.get("eps_ttm"), "pe_ttm": cv.get("pe_ttm_computed") or cv.get("pe_ttm_api"),
              "pb_now": cv.get("pb_now"), "bvps_latest": (ctx.get("balance_latest") or {}).get("bvps")})
    # first-to-last move of every bound series, for so_what templates ({nim_move}, ...)
    for mid, b in (variant.get("metrics") or {}).items():
        s = actual_series(ctx, b["series"])
        vals = list(s.values())
        if len(vals) >= 2:
            f[f"{mid}_move"] = abs(vals[-1] - vals[0])
            f[f"{mid}_range"] = max(vals) - min(vals)
    growth = actual_series(ctx, "Net Profit Growth")
    f["growth_observations"] = len(growth)
    f["downturn_years"] = sum(1 for v in growth.values() if v < 0)
    gv = list(growth.values())
    if len(gv) >= 3:
        earlier = statistics.median(gv[:-1])
        f["profit_growth_latest"], f["profit_growth_earlier_median"] = gv[-1], earlier
        f["profit_growth_slowing"] = gv[-1] < earlier
    pbh = [r["pb"] for r in ctx["fy_series"].values() if r.get("pb")]
    now = cv.get("pb_now")
    if now and pbh:
        f["pb_vs_history"] = "above" if now > max(pbh) else ("below" if now < min(pbh) else "within")
    pb = (ctx["peers"].get("bank") or {})
    med = (pb.get("median_excl_subject") or {}).get("pb_now")
    if now and med:
        f["pb_vs_peers"] = "above" if now > 1.10 * med else ("below" if now < 0.90 * med else "within")
        f["peer_pb_median"] = med
    merger = [y for y, v in actual_series(ctx, "Advances Growth").items() if v > (variant["metrics"]["advances_growth"].get("merger_cap_pct") or 1e9)]
    f["merger_years"] = [fy(y) for y in merger]
    return f


# ----------------------------------------------------------------------------------------------
# chips
# ----------------------------------------------------------------------------------------------
def _decimals_text(chip: dict, decimals: int) -> dict:
    if chip and chip["unit"] == "%":
        chip["value_text"] = f"{chip['value']:.{decimals}f}%"
    return chip


def build_chips(ctx: dict, variant: dict, facts: dict, cases: list[dict]) -> dict[str, dict]:
    chips: dict[str, dict] = {}
    latest = ctx["meta"]["latest_actual_year"]
    per = fy(latest)
    peer_med = (ctx["peers"].get("bank") or {}).get("median_excl_subject") or {}
    for mid, b in (variant.get("metrics") or {}).items():
        s = actual_series(ctx, b["series"])
        if latest not in s:
            continue
        hist = list(s.values())
        extra = None
        cap = b.get("merger_cap_pct")
        if cap is not None:
            kept = {y: v for y, v in s.items() if v <= cap}
            dropped = [fy(y) for y in s if y not in kept]
            if dropped:
                hist = list(kept.values())
                extra = f"{', '.join(dropped)} left out of the range (merger-affected, not like-for-like)."
        pm = peer_med.get(b["peer"]) if b.get("peer") else None
        c = ve.evaluate(mid, s[latest], hist, pm, facts, per, calc_extra=extra)
        if c:
            chips[mid] = _decimals_text(c, b.get("decimals", 1))
    cv = ctx["current_valuation"]
    pb_hist = [r["pb"] for r in ctx["fy_series"].values() if r.get("pb")]
    if cv.get("pb_now"):
        chips_pb = ve.evaluate("pb", cv["pb_now"], pb_hist, peer_med.get("pb_now"), facts, "snapshot")
        if chips_pb:
            chips["pb"] = chips_pb
    pe_hist = [r["pe"] for r in ctx["fy_series"].values() if r.get("pe")]
    pe_now = cv.get("pe_ttm_computed") or cv.get("pe_ttm_api")
    if pe_now:
        c = ve.evaluate("pe", pe_now, pe_hist, None, facts, "TTM")
        if c:
            chips["pe"] = c
    if cases and facts.get("price"):
        c = ve.fair_value(facts["price"], [(x["low_price"], x["high_price"]) for x in cases], "snapshot")
        if c:
            chips["fair_value"] = c
    return chips


# ----------------------------------------------------------------------------------------------
# fair-value cases
# ----------------------------------------------------------------------------------------------
def fair_value_cases(ctx: dict, variant: dict, facts: dict) -> list[dict]:
    price = facts.get("price")
    if not price:
        return []
    fys = ctx["fy_series"]
    peers = (ctx["peers"].get("bank") or {}).get("rows") or {}
    subject = ctx["meta"]["ticker"].upper()
    nxt = next(iter(sorted(ctx["model_forecast"])), None)
    bases = {"bvps": (facts.get("bvps_latest"), "book value per share"),
             "eps_ttm": (facts.get("eps_ttm"), "TTM earnings per share"),
             "eps_next": ((ctx["model_forecast"].get(nxt) or {}).get("eps") if nxt else None, f"{fy(nxt)} EPS forecast" if nxt else "EPS forecast")}
    sources = {"pb_history": [r["pb"] for r in fys.values() if r.get("pb")],
               "pe_history": [r["pe"] for r in fys.values() if r.get("pe")],
               "peer_pb": [r.get("pb_now") if r.get("pb_now") is not None else r.get("pb_fy_avg") for t, r in peers.items() if t != subject]}
    out = []
    for spec in variant.get("fair_value_cases") or []:
        base, base_name = bases.get(spec["base"], (None, spec["base"]))
        lo = quantile(sources.get(spec["low"]["source"], []), spec["low"]["stat"])
        hi = quantile(sources.get(spec["high"]["source"], []), spec["high"]["stat"])
        if not base or lo is None or hi is None:
            continue
        lo_p, hi_p = lo * base, hi * base
        out.append({"key": spec["key"], "label": spec["label"], "basis": spec["basis"], "base_name": base_name, "base_value": base,
                    "low_mult": min(lo, hi), "high_mult": max(lo, hi), "low_price": min(lo_p, hi_p), "high_price": max(lo_p, hi_p),
                    "vs_price_pct_low": (min(lo_p, hi_p) / price - 1) * 100, "vs_price_pct_high": (max(lo_p, hi_p) / price - 1) * 100})
    return out


def check_cases(cases: list[dict], price: Optional[float]) -> list[str]:
    """Bank equivalent of the fair-value recompute check (validate.py recomputes EBITDA-based cases only)."""
    errs = []
    for c in cases:
        if abs(c["low_mult"] * c["base_value"] - c["low_price"]) > 0.01 * max(abs(c["low_price"]), 1) or \
           abs(c["high_mult"] * c["base_value"] - c["high_price"]) > 0.01 * max(abs(c["high_price"]), 1):
            errs.append(f"Fair-value case '{c['label']}' does not recompute from its multiple and {c['base_name']}.")
        if price and abs((c["low_price"] / price - 1) * 100 - c["vs_price_pct_low"]) > 0.5:
            errs.append(f"Fair-value case '{c['label']}' is not measured against the snapshot price.")
    return errs


# ----------------------------------------------------------------------------------------------
# signposts and lights
# ----------------------------------------------------------------------------------------------
def _status(cur: float, thr: float, higher_better: bool, near: float = 0.15) -> str:
    if higher_better:
        return "good" if cur >= thr else ("neutral" if cur >= thr * (1 - near) else "concern")
    return "good" if cur <= thr else ("neutral" if cur <= thr * (1 + near) else "concern")


def build_signposts(ctx: dict, variant: dict, facts: dict) -> list[dict]:
    out = []
    latest = ctx["meta"]["latest_actual_year"]
    for spec in variant.get("signposts") or []:
        s = actual_series(ctx, spec["series"])
        if latest not in s or len(s) < 3:
            continue
        step = spec.get("round", 1.0)
        thr = round(statistics.median(s.values()) / step) * step
        dec = 1 if step < 1 else 0
        floor = spec["direction"] == "floor"
        cur = s[latest]
        unit = spec.get("unit", "")
        out.append({"metric": spec["metric"], "current": cur, "current_text": f"{cur:.{spec.get('decimals', 1)}f}{unit} ({fy(latest)})",
                    "threshold": thr, "threshold_text": f"{'Above' if floor else 'Below'} {thr:.{dec}f}{unit}",
                    "status": _status(cur, thr, floor), "why": spec["why"]})
    return out[:5]


def build_lights(variant: dict, chips: dict, tests: list[dict]) -> list[dict]:
    names = {"business": "Business health", "profit": "Profit trend", "price": "Price vs value", "momentum": "Momentum"}
    word = {"good": "good", "neutral": "neutral", "concern": "a concern"}
    lights = {}
    reasons = {}
    for key, rule in (variant.get("lights") or {}).items():
        vs = {c: chips[c]["verdict"] for c in rule["chips"] if c in chips}
        concerns = [c for c, v in vs.items() if v == "concern"]
        goods = [c for c, v in vs.items() if v == "good"]
        if any(c in concerns for c in rule.get("concern_if_any", [])) or len(concerns) >= rule.get("concern_count", 99):
            lights[key] = "concern"
        elif not concerns and len(goods) >= rule.get("good_min", 99):
            lights[key] = "good"
        else:
            lights[key] = "neutral"
        reasons[key] = rule["reason"].format_map({c: word.get(vs.get(c), "n/a") for c in rule["chips"]})
    tv = {t["key"]: t["verdict"] for t in tests}
    lights["momentum"] = ("concern" if tv.get("trend") == "concern" else
                          "good" if tv.get("trend") == "good" and tv.get("momentum") == "good" and tv.get("stretch") != "concern" else "neutral")
    reasons["momentum"] = "; ".join(f"{t['label'].lower()} {word[t['verdict']]}" for t in tests) + "." if tests else "Price history not available."
    return [{"key": k, "label": names[k], "light": lights.get(k, "neutral"), "icon": ve.ICON[lights.get(k, "neutral")], "reason": reasons.get(k, "")}
            for k in ("business", "profit", "price", "momentum")]
