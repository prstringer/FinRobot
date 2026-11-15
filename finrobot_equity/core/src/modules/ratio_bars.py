#!/usr/bin/env python
# coding: utf-8
"""
Per-ratio bars with benchmarks and verdict chips (Part E), replacing the single-scale radar chart.

Each ratio keeps its own scale. For every available ratio the facts hold the current value, its own five-year range,
the peer median (when supplied), the direction (higher or lower is better) and a verdict with its yardstick.

Verdict rules (no verdict without a yardstick):
  1. peer median supplied: compare with peers (within +/-10% of the median is neutral),
  2. else three or more own years: position inside the own five-year range (best third good, worst third concern),
  3. else a general norm where one exists (labelled "general norm, not company-specific"),
  4. else the verdict is None.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import pandas as pd  # noqa: E402

GOOD, NEUTRAL, CONCERN = "#2f855a", "#b7791f", "#c53030"
CHIP = {"good": GOOD, "neutral": NEUTRAL, "concern": CONCERN}
PEER_TOLERANCE = 0.10

# key -> (label, ratios_df column, scale to display units, unit, higher_is_better, general norm or None)
# general norm: (good_if, concern_if) in display units, interpreted by direction
METRICS = {
    "gross_margin": ("Gross margin", "grossProfitMargin", 100.0, "%", True, None),
    "ebitda_margin": ("EBITDA margin", "ebitdaMargin", 100.0, "%", True, None),
    "net_margin": ("Net margin", "netProfitMargin", 100.0, "%", True, None),
    "roe": ("Return on equity", "returnOnEquity", 100.0, "%", True, None),
    "roa": ("Return on assets", "returnOnAssets", 100.0, "%", True, None),
    "current_ratio": ("Current ratio", "currentRatio", 1.0, "x", True, (1.5, 1.0)),
    "interest_coverage": ("Interest coverage", "interestCoverage", 1.0, "x", True, (4.0, 2.5)),
    "debt_equity": ("Debt to equity", "debtEquityRatio", 1.0, "x", False, (0.5, 1.0)),
    "sga_pct": ("SG&A % of revenue", None, 100.0, "%", False, None),
}

SO_WHAT = {
    "gross_margin": {"good": "Pricing and product mix are holding up, which protects profit when costs rise.",
                     "neutral": "Pricing power is neither improving nor eroding, so profit depends on cost control.",
                     "concern": "Pricing or mix is weakening, which squeezes profit before overheads are counted."},
    "ebitda_margin": {"good": "Operations are turning a healthy share of sales into profit.",
                      "neutral": "Operating profitability is middling, leaving little cushion for a cost shock.",
                      "concern": "Operations keep little of each sale as profit, so earnings are fragile."},
    "net_margin": {"good": "A healthy share of sales survives interest and tax as profit for owners.",
                   "neutral": "Profit left for owners is modest and sensitive to interest and tax.",
                   "concern": "Very little of each sale reaches owners, so small shocks can erase profit."},
    "roe": {"good": "Owners' money is earning a strong return.",
            "neutral": "Owners' money earns an ordinary return.",
            "concern": "Owners' money earns a weak return, which does not justify a premium price."},
    "roa": {"good": "The asset base is generating profit efficiently.",
            "neutral": "Assets earn an ordinary return.",
            "concern": "Assets earn little, so the business is capital heavy for its profit."},
    "current_ratio": {"good": "Short-term assets comfortably cover short-term bills.",
                      "neutral": "Short-term liquidity is adequate but not generous.",
                      "concern": "Short-term assets barely cover short-term bills, so liquidity is tight."},
    "interest_coverage": {"good": "Profit covers interest payments with room to spare.",
                          "neutral": "Profit covers interest, but a profit dip would make debt costs felt.",
                          "concern": "Profit covers interest only thinly, so a downturn puts debt service at risk."},
    "debt_equity": {"good": "Borrowing is modest compared with owners' funds, which lowers risk.",
                    "neutral": "Borrowing is moderate compared with owners' funds.",
                    "concern": "Borrowing relative to owners' funds is above the benchmark, which adds risk if earnings fall."},
    "sga_pct": {"good": "Overheads take a small share of sales, which leaves more for profit.",
                "neutral": "Overheads take a typical share of sales.",
                "concern": "Overheads take a growing share of sales and are the main drag on profit."},
}


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _year_col(df: pd.DataFrame) -> str:
    return "calendarYear" if "calendarYear" in df.columns else "year"


def _series(ratios_df: pd.DataFrame, col: str, scale: float, years: int) -> list[tuple[int, float]]:
    ycol = _year_col(ratios_df)
    out = []
    for _, r in ratios_df.sort_values(ycol).iterrows():
        v = _f(r.get(col)) if col in ratios_df.columns else None
        if v is not None:
            out.append((int(r[ycol]), v * scale))
    return out[-years:]


def _verdict(key: str, value: float, rng: tuple[float, float], n_years: int, peer_median: Optional[float],
             higher_better: bool, norm) -> dict:
    """Pick the yardstick and the verdict; return {verdict, yardstick, source_note, basis}."""
    sign = 1 if higher_better else -1
    if peer_median is not None and peer_median != 0:
        rel = sign * (value - peer_median) / abs(peer_median)
        v = "good" if rel > PEER_TOLERANCE else ("concern" if rel < -PEER_TOLERANCE else "neutral")
        return {"verdict": v, "yardstick": "peers", "source_note": "report data (peer median, latest year)"}
    lo, hi = rng
    if n_years >= 3 and hi > lo:
        pos = (value - lo) / (hi - lo)
        better = pos if higher_better else 1 - pos
        v = "good" if better >= 2 / 3 else ("concern" if better <= 1 / 3 else "neutral")
        return {"verdict": v, "yardstick": "own_history", "source_note": f"report data (own {n_years}-year range)"}
    if norm is not None:
        good_at, concern_at = norm
        if higher_better:
            v = "good" if value >= good_at else ("concern" if value < concern_at else "neutral")
        else:
            v = "good" if value <= good_at else ("concern" if value > concern_at else "neutral")
        return {"verdict": v, "yardstick": "general_norm", "source_note": "general norm, not company-specific"}
    return {"verdict": None, "yardstick": None, "source_note": "no yardstick available"}


def compute_ratio_bar_facts(ratios_df: Optional[pd.DataFrame], key_metrics: Optional[pd.DataFrame] = None,
                            peer_medians: Optional[dict] = None, income_df: Optional[pd.DataFrame] = None,
                            years: int = 5) -> dict:
    """Facts for each available ratio, keyed by metric key, in a stable display order.

    ``peer_medians`` maps a metric key (see ``METRICS``) to the peer median in display units (percent for margins,
    ROE and ROA; times for the rest). ``income_df`` is used only for SG&A % of revenue. ``key_metrics`` is accepted for
    forward compatibility (ROE falls back to its ``roe`` column when the ratios table lacks it).
    """
    peer_medians = peer_medians or {}
    facts: dict[str, dict] = {}
    for key, (label, col, scale, unit, higher, norm) in METRICS.items():
        if key == "sga_pct":
            ser = []
            if income_df is not None and not income_df.empty and {"revenue", "sellingGeneralAndAdministrativeExpenses"} <= set(income_df.columns):
                inc = income_df.sort_values("year")
                ser = [(int(r["year"]), r["sellingGeneralAndAdministrativeExpenses"] / r["revenue"] * 100)
                       for _, r in inc.iterrows() if _f(r["revenue"]) and _f(r["sellingGeneralAndAdministrativeExpenses"]) is not None][-years:]
        else:
            ser = _series(ratios_df, col, scale, years) if ratios_df is not None and not ratios_df.empty else []
            if not ser and key == "roe" and key_metrics is not None and not key_metrics.empty and "roe" in key_metrics.columns:
                ser = _series(key_metrics, "roe", scale, years)
        if not ser:
            continue
        vals = [v for _, v in ser]
        value, rng = vals[-1], (min(vals), max(vals))
        pm = _f(peer_medians.get(key))
        vd = _verdict(key, value, rng, len(vals), pm, higher, norm)
        so_what = SO_WHAT[key].get(vd["verdict"]) if vd["verdict"] else "Not enough history or benchmark to judge this ratio."
        pos = (value - rng[0]) / (rng[1] - rng[0]) * 100 if rng[1] > rng[0] else None
        facts[key] = {
            "label": label, "unit": unit, "higher_is_better": higher,
            "direction": "higher is better" if higher else "lower is better",
            "latest_year": ser[-1][0], "history": {y: v for y, v in ser},
            "verdict": vd["verdict"], "yardstick": vd["yardstick"], "so_what": so_what, "source_note": vd["source_note"],
            "scale": {"value": value, "range_min": rng[0], "range_max": rng[1], "peer_median": pm},
            "position_in_own_range_pct": pos,
        }
    return facts


def peer_medians_from_ratios(peer_ratios: dict[str, pd.DataFrame], subject: str, income_by_peer: Optional[dict] = None) -> dict:
    """Latest-year peer median per metric, excluding the subject. ``peer_ratios`` maps ticker -> ratios DataFrame."""
    out: dict[str, float] = {}
    for key, (label, col, scale, unit, higher, norm) in METRICS.items():
        vals = []
        for t, df in peer_ratios.items():
            if t.upper() == subject.upper() or df is None or df.empty:
                continue
            if key == "sga_pct":
                inc = (income_by_peer or {}).get(t)
                if inc is None or inc.empty or "sellingGeneralAndAdministrativeExpenses" not in inc.columns:
                    continue
                r = inc.sort_values("year").iloc[-1]
                if _f(r["revenue"]) and _f(r["sellingGeneralAndAdministrativeExpenses"]) is not None:
                    vals.append(r["sellingGeneralAndAdministrativeExpenses"] / r["revenue"] * 100)
            else:
                s = _series(df, col, scale, 1)
                if s:
                    vals.append(s[-1][1])
        if vals:
            out[key] = float(pd.Series(vals).median())
    return out


def ratio_bar_rows(facts: dict) -> list[dict]:
    """Render-ready rows (HTML or PDF): label, value text, range text, peer text, chip, so_what."""
    rows = []
    for key, f in facts.items():
        u, s = f["unit"], f["scale"]
        fmt = (lambda v: f"{v:.1f}%") if u == "%" else (lambda v: f"{v:.2f}x")
        rows.append({
            "key": key, "label": f"{f['label']} ({f['direction']})", "value": fmt(s["value"]),
            "own_range": f"{fmt(s['range_min'])} to {fmt(s['range_max'])}",
            "peer_median": fmt(s["peer_median"]) if s["peer_median"] is not None else "n/a",
            "verdict": f["verdict"], "yardstick": f["yardstick"], "so_what": f["so_what"], "source_note": f["source_note"],
        })
    return rows


def ratio_bar_text_lines(facts: dict) -> list[str]:
    lines = ["RATIO BENCHMARKS (value; own five-year range; peer median; verdict vs yardstick):"]
    for r in ratio_bar_rows(facts):
        v = f"{r['verdict']} vs {r['yardstick'].replace('_', ' ')}" if r["verdict"] else "no verdict (no yardstick)"
        lines.append(f"  {r['label']}: {r['value']}; range {r['own_range']}; peer median {r['peer_median']}; {v}. "
                     f"{r['so_what']} ({r['source_note']})")
    return lines


def draw_ratio_bars(facts: dict, out_path: str, title: str = "Key ratios against their own history and peers") -> Optional[str]:
    """One horizontal bar per ratio: own range (grey band), current value (dot, coloured by verdict), peer median (diamond)."""
    if not facts:
        return None
    n = len(facts)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 0.62 * n + 1.1), squeeze=False)
    for ax, (key, f) in zip(axes[:, 0], facts.items()):
        s = f["scale"]
        vals = [s["value"], s["range_min"], s["range_max"]] + ([s["peer_median"]] if s["peer_median"] is not None else [])
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.18 or max(abs(hi) * 0.1, 0.5)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-1, 1)
        ax.barh(0, s["range_max"] - s["range_min"], left=s["range_min"], height=0.7, color="#e2e8f0", edgecolor="#cbd5e1")
        color = CHIP.get(f["verdict"], "#64748b")
        ax.plot([s["value"]], [0], marker="o", markersize=9, color=color, zorder=3)
        if s["peer_median"] is not None:
            ax.plot([s["peer_median"]], [0], marker="D", markersize=7, color="#1a365d", zorder=2)
        txt = f"{s['value']:.1f}%" if f["unit"] == "%" else f"{s['value']:.2f}x"
        ax.text(1.01, 0.5, txt, transform=ax.transAxes, va="center", fontsize=8, fontweight="bold")
        chip = f["verdict"].capitalize() if f["verdict"] else "No verdict"
        ax.text(1.12, 0.5, chip, transform=ax.transAxes, va="center", ha="left", fontsize=8, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="none"))
        ax.set_yticks([]); ax.tick_params(axis="x", labelsize=7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"{f['label']} ({f['direction']})", loc="left", fontsize=8, pad=2)
    handles = [mpatches.Patch(facecolor="#e2e8f0", edgecolor="#cbd5e1", label="Own range (up to 5 years)"),
               plt.Line2D([], [], marker="o", color="#64748b", linestyle="", label="Latest value (colour = verdict)"),
               plt.Line2D([], [], marker="D", color="#1a365d", linestyle="", label="Peer median")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.suptitle(title, fontsize=10, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 0.88, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
