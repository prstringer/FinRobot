#!/usr/bin/env python
# coding: utf-8
"""
Fiscal calendar and reporting-basis guards for Indian filings.

* Indian corporate fiscal year: 1 April to 31 March. "FY26" is the year ended 31 March 2026 (1 Apr 2025 to 31 Mar 2026).
  Full-year columns carry the "A" suffix; the trailing four quarters are a separate TTM figure, never a substitute for an
  "A" column. ``check_fiscal_year_ends`` fails loudly if a column labelled as a fiscal year does not end on 31 March.
* Reporting tiers: standalone and consolidated filings are never mixed. All structured figures in a report come from ONE
  basis (the one the pipeline chose). ``quarter_basis_comparison`` computes the latest quarter's PAT growth on BOTH bases
  so a headline quoting the other basis (for example "net profit up 83%" on a standalone filing) cannot be read as the
  consolidated trend.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

FY_START_MONTH, FY_END_MONTH, FY_END_DAY = 4, 3, 31
DIVERGENCE_POINTS = 10.0       # standalone vs consolidated YoY gap (percentage points) worth a warning


def fy_label(year_or_label) -> str:
    """'2026A' or 2026 -> 'FY26'; '2027E' -> 'FY27E'."""
    m = re.fullmatch(r"(\d{4})([AE]?)", str(year_or_label))
    return f"FY{m.group(1)[2:]}{'E' if m.group(2) == 'E' else ''}" if m else str(year_or_label)


def fy_span(year: int) -> str:
    """Human description of an Indian fiscal year: FY26 -> '1 Apr 2025 to 31 Mar 2026'."""
    return f"1 Apr {year - 1} to 31 Mar {year}"


def check_fiscal_year_ends(frames: dict[str, Optional[pd.DataFrame]]) -> list[str]:
    """Issues for any annual row whose period end is not 31 March (empty list means the calendar is consistent).

    ``frames`` maps a name ('income', 'balance', 'cash') to a normalised annual DataFrame with ``date`` and ``year``.
    The fiscal year number must also equal the calendar year of the period end (FY26 ends in 2026).
    """
    issues = []
    for name, df in frames.items():
        if df is None or df.empty or "date" not in df.columns:
            continue
        for _, r in df.iterrows():
            d = pd.Timestamp(r["date"])
            if (d.month, d.day) != (FY_END_MONTH, FY_END_DAY):
                issues.append(f"{name}: fiscal year {int(r['year'])} ends {d.date()}, not 31 March")
            elif int(r["year"]) != d.year:
                issues.append(f"{name}: fiscal year {int(r['year'])} does not match period end {d.date()}")
    return issues


def _quarter_pat(client, symbol: str, statement_type: str):
    from .finedge_api import normalize_income_statement
    rows = client.financials(symbol, "pl", "quarterly", statement_type)
    if not rows:
        return None
    df = normalize_income_statement(rows, "quarterly")
    if df.empty:
        return None
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    latest = df.iloc[0]
    prior = df[(df["date"] <= latest["date"] - pd.Timedelta(days=350)) & (df["date"] >= latest["date"] - pd.Timedelta(days=380))]
    if prior.empty:
        return None
    p0, p1 = float(latest["netIncome"]), float(prior.iloc[0]["netIncome"])
    return {"quarter_end": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"), "pat_cr": p0 / 1e7, "pat_prior_cr": p1 / 1e7,
            "yoy_pct": (p0 / p1 - 1) * 100 if p1 else None, "revenue_cr": float(latest["revenue"]) / 1e7}


def quarter_basis_comparison(client, symbol: str, used: str = "c") -> Optional[dict]:
    """Latest-quarter PAT and YoY growth on both bases. ``used`` is the basis every structured figure comes from."""
    from .finedge_api import FinEdgeError
    out: dict = {"used": "consolidated" if used == "c" else "standalone"}
    for key, st in (("consolidated", "c"), ("standalone", "s")):
        try:
            out[key] = _quarter_pat(client, symbol, st)
        except FinEdgeError:
            out[key] = None
    c, s = out.get("consolidated"), out.get("standalone")
    if not c and not s:
        return None
    out["diverges"] = bool(c and s and c["yoy_pct"] is not None and s["yoy_pct"] is not None
                           and abs(c["yoy_pct"] - s["yoy_pct"]) >= DIVERGENCE_POINTS)
    return out


def basis_fact_block(basis: dict, fy_year: Optional[int] = None) -> str:
    """Facts for the text layer: one basis for every figure, and the other basis stated where it differs."""
    L = ["REPORTING BASIS (computed):",
         f"  Every structured figure in this report (annual, TTM, balance sheet, ratios) is on the {basis['used'].upper()} basis. "
         "The Indian fiscal year runs 1 April to 31 March"
         + (f"; FY{str(fy_year)[2:]} is {fy_span(fy_year)}." if fy_year else ".")]
    for key in ("consolidated", "standalone"):
        q = basis.get(key)
        if q and q.get("yoy_pct") is not None:
            L.append(f"  Latest quarter ({q['quarter_end']}), {key}: PAT Rs {q['pat_cr']:.2f} Cr against Rs {q['pat_prior_cr']:.2f} Cr a year earlier ({q['yoy_pct']:+.0f}% YoY).")
    if basis.get("diverges"):
        c, s = basis["consolidated"], basis["standalone"]
        L.append(f"  The two bases disagree by {abs(c['yoy_pct'] - s['yoy_pct']):.0f} points. A headline quoting quarterly profit growth may be "
                 f"on the other basis. State quarterly PAT growth only with its basis, using the figures above, and never mix a "
                 f"{'standalone' if basis['used'] == 'consolidated' else 'consolidated'} growth rate into {basis['used']} trends.")
    return "\n".join(L)


def basis_warnings(basis: Optional[dict], fiscal_issues: list[str]) -> list[str]:
    out = [f"Fiscal calendar: {i}." for i in fiscal_issues]
    if basis and basis.get("diverges"):
        c, s = basis["consolidated"], basis["standalone"]
        out.append(f"Standalone and consolidated results differ for the quarter to {c['quarter_end']}: PAT growth is {s['yoy_pct']:+.0f}% "
                   f"standalone but {c['yoy_pct']:+.0f}% consolidated. Figures in this report are {basis['used']}; headlines may quote the other basis.")
    return out
