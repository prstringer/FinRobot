#!/usr/bin/env python
# coding: utf-8
"""
Bank report template: which rows of the bank analysis table (built by
``finedge_api.build_bank_metrics_table``) go into each report table, the
section titles to use, and helpers shared by the HTML and PDF reports.
"""

import pandas as pd


def is_bank_analysis(analysis_df: pd.DataFrame, analysis_summary: dict | None = None) -> bool:
    """True for bank analysis tables (template 'bank' or a 'Net Interest Income' row)."""
    if analysis_summary and analysis_summary.get("template") == "bank":
        return True
    return (analysis_df is not None and not analysis_df.empty and "metrics" in analysis_df.columns
            and bool((analysis_df["metrics"] == "Net Interest Income").any()))


def bank_balance_table(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Returns, asset quality and balance-sheet rows in a fixed order (actual years only)."""
    rows = BANK_QUALITY_METRICS + BANK_BALANCE_METRICS
    cols = ["metrics"] + [c for c in analysis_df.columns if isinstance(c, str) and c.endswith("A")]
    df = analysis_df[analysis_df["metrics"].isin(rows)][cols].copy()
    df["_order"] = df["metrics"].map({m: i for i, m in enumerate(rows)})
    return df.sort_values("_order").drop(columns="_order")


# Bank template: which rows of the bank analysis table go into each report table
BANK_SUMMARY_METRICS = ["Net Interest Income", "NII Growth", "Other Income", "Total Income", "Operating Expenses",
                        "Pre-Provision Profit", "Provisions", "Net Profit", "Net Profit Growth"]
BANK_QUALITY_METRICS = ["NIM", "ROA", "ROE", "Cost-to-Income", "Credit Cost", "Gross NPA", "Net NPA", "CET1 Ratio"]
BANK_BALANCE_METRICS = ["Advances", "Advances Growth", "Deposits", "Deposits Growth", "Credit-Deposit Ratio",
                        "Book Value per Share", "PB Ratio"]
BANK_REPORT_LABELS = {
    "revenue_chart_title": "Total Income & Pre-Provision Profit",
    "revenue_section_title": "Income & Pre-Provision Profit",
    "peer_chart_title": "P/B Peer Comparison",
    "credit_section_title": "Returns, Asset Quality & Balance Sheet",
    "peer_valuation_title": "Peer Banks (latest fiscal year)",
    "peer_ebitda_title": "Peer EBITDA (not meaningful for banks)",
    "peer_ev_ebitda_title": "Peer Banks (latest fiscal year)",
}


def _latest_numeric(analysis_df: pd.DataFrame, metric: str):
    """Latest actual-year numeric value of a metric row, or None."""
    row = analysis_df[analysis_df["metrics"] == metric]
    if row.empty:
        return None
    years = sorted(c for c in analysis_df.columns if isinstance(c, str) and c.endswith("A"))
    for y in reversed(years):
        v = pd.to_numeric(row[y].iloc[0], errors="coerce")
        if pd.notna(v):
            return float(v)
    return None


def _history_numeric(analysis_df: pd.DataFrame, metric: str) -> list:
    row = analysis_df[analysis_df["metrics"] == metric]
    if row.empty:
        return []
    years = sorted(c for c in analysis_df.columns if isinstance(c, str) and c.endswith("A"))
    vals = pd.to_numeric(row[years].iloc[0], errors="coerce")
    return [float(v) for v in vals if pd.notna(v)]


def format_bank_peer_table(peer_df: pd.DataFrame) -> pd.DataFrame:
    """Display table for peer banks: tickers as rows, ratios as formatted columns."""
    if peer_df is None or peer_df.empty:
        return pd.DataFrame()
    out = peer_df.set_index("ticker")
    display = pd.DataFrame(index=out.index)
    display.index.name = "Bank"
    display["FY"] = out["year"].astype(int).astype(str)
    display["P/B (x)"] = out["P/B"].map(lambda v: f"{v:.2f}x" if pd.notna(v) else "N/A")
    for col, digits in [("ROE", 1), ("ROA", 2), ("NIM", 2), ("GNPA", 2), ("NNPA", 2), ("CET1", 1)]:
        display[col] = out[col].map(lambda v, d=digits: f"{v * 100:.{d}f}%" if pd.notna(v) else "N/A")
    return display
