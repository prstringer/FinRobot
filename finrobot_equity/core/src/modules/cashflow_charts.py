#!/usr/bin/env python
# coding: utf-8
"""Cash-flow charts (Part D3): EBITDA vs CFO with conversion %, a cash waterfall, and working-capital days.

All functions take the dict from ``cashflow_analysis.compute_cashflow_facts`` and write a PNG, returning its path
(or None when the inputs are missing). Years always run oldest to newest, left to right. Values are in rupee crore.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRIMARY, ACCENT, GOOD, BAD, GRAY = "#1a365d", "#d4a843", "#2f855a", "#c53030", "#94a3b8"


def _labels(facts: dict, with_ttm: bool = True) -> list[str]:
    labs = list(facts.get("years", []))
    if with_ttm and facts.get("ttm_label"):
        labs.append(facts["ttm_label"])
    return labs


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def ebitda_vs_cfo_chart(facts: dict, out_path: str, ticker: str = "") -> Optional[str]:
    """Paired bars of EBITDA and operating cash flow per year, with CFO/EBITDA labelled above each pair."""
    if not facts.get("available"):
        return None
    labs = [l for l in _labels(facts) if facts["per_year"][l]["cfo"] is not None and facts["per_year"][l]["ebitda"] is not None]
    if not labs:
        return None
    py = facts["per_year"]
    x = np.arange(len(labs)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, [py[l]["ebitda"] for l in labs], w, color=PRIMARY, label="EBITDA")
    ax.bar(x + w / 2, [py[l]["cfo"] for l in labs], w, color=ACCENT, label="Operating cash flow")
    top = max(max(py[l]["ebitda"], py[l]["cfo"]) for l in labs)
    for i, l in enumerate(labs):
        c = py[l]["conversion"]
        if c is not None:
            ax.text(i, max(py[l]["ebitda"], py[l]["cfo"]) + top * 0.03, f"{c * 100:.0f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=GOOD if c >= 0.8 else (BAD if c < 0.5 else "#475569"))
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("Rs crore")
    ax.set_title(f"{ticker + ' - ' if ticker else ''}EBITDA vs operating cash flow (% = conversion)", fontsize=9.5, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    return _save(fig, out_path)


def cash_waterfall_chart(facts: dict, out_path: str, year: Optional[str] = None, ticker: str = "") -> Optional[str]:
    """Waterfall for one year: CFO, capex, FCF (subtotal), dividends, buybacks, net debt, other (balancing), change in cash."""
    if not facts.get("available"):
        return None
    year = year or facts["years"][-1]
    r = facts["per_year"].get(year)
    if not r or r["cfo"] is None or r["capex"] is None or r["net_change"] is None:
        return None
    steps = [("Operating\ncash flow", r["cfo"], "delta"), ("Capex", r["capex"], "delta"), ("Free cash\nflow", None, "subtotal")]
    running = r["cfo"] + r["capex"]
    for lab, key in (("Dividends", "dividends"), ("Buybacks", "buybacks"), ("Net debt\nraised/(repaid)", "debt_net")):
        if r[key] is not None:
            steps.append((lab, r[key], "delta")); running += r[key]
    other = r["net_change"] - running
    steps.append(("Other flows\n(balancing)", other, "delta"))
    steps.append(("Change\nin cash", None, "total"))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    level, xs = 0.0, np.arange(len(steps))
    for i, (lab, v, kind) in enumerate(steps):
        if kind == "delta":
            bottom = level if v >= 0 else level + v
            ax.bar(i, abs(v), bottom=bottom, color=GOOD if v >= 0 else BAD, width=0.6)
            ax.text(i, level + v, f"{v:+,.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
            level += v
        else:
            val = level if kind == "subtotal" else r["net_change"]
            ax.bar(i, val, color=PRIMARY if kind == "subtotal" else GRAY, width=0.6)
            ax.text(i, val, f"{val:,.0f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8, fontweight="bold")
            if kind == "subtotal":
                level = val
    ax.axhline(0, color="#334155", linewidth=0.8)
    levels, run = [0.0], 0.0
    for _, v, kind in steps:
        if kind == "delta":
            run += v
        elif kind == "subtotal":
            run = r["cfo"] + r["capex"]
        levels.append(run)
    levels.append(r["net_change"])
    lo, hi = min(levels), max(levels)
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - span * 0.1, hi + span * 0.1)
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in steps], fontsize=8)
    ax.set_ylabel("Rs crore")
    ax.set_title(f"{ticker + ' - ' if ticker else ''}Where operating cash went, {year}", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    return _save(fig, out_path)


def working_capital_days_chart(facts: dict, out_path: str, ticker: str = "") -> Optional[str]:
    """Lines for inventory, receivable and payable days, plus the cash cycle, oldest to newest."""
    wc = facts.get("working_capital_days") or {}
    labs = [l for l in facts.get("years", []) if l in wc]
    if len(labs) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, name, color, style in (("inventory_days", "Inventory days", PRIMARY, "-"), ("receivable_days", "Receivable days", ACCENT, "-"),
                                    ("payable_days", "Payable days", GRAY, "-"), ("cash_conversion_days", "Cash cycle (days)", BAD, "--")):
        vals = [wc[l][key] for l in labs]
        ax.plot(labs, vals, marker="o", color=color, linestyle=style, linewidth=2, label=name)
        ax.annotate(f"{vals[-1]:.0f}", (len(labs) - 1, vals[-1]), textcoords="offset points", xytext=(6, 0), fontsize=8, color=color)
    ax.set_ylabel("Days")
    ax.set_title(f"{ticker + ' - ' if ticker else ''}Working-capital days (year-end balances)", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    return _save(fig, out_path)


def draw_cashflow_charts(facts: dict, out_dir: str, ticker: str = "") -> dict:
    """All three charts; returns {name: path or None}."""
    t = ticker or "report"
    return {
        "ebitda_vs_cfo": ebitda_vs_cfo_chart(facts, os.path.join(out_dir, f"{t}_ebitda_vs_cfo_chart.png"), ticker),
        "cash_waterfall": cash_waterfall_chart(facts, os.path.join(out_dir, f"{t}_cash_waterfall_chart.png"), ticker=ticker),
        "working_capital_days": working_capital_days_chart(facts, os.path.join(out_dir, f"{t}_working_capital_days_chart.png"), ticker),
    }
