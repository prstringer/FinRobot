#!/usr/bin/env python
# coding: utf-8
"""
Cash-flow facts, diagnostics and verdicts, computed in code (Part D).

Inputs are the normalised DataFrames from ``financial_data`` (absolute currency units; one row per fiscal year):

* ``income_statement``: year, revenue, ebitda, costOfRevenue
* ``balance_sheet``:    year, inventory, netReceivables, accountPayables
* ``cash_flow``:        year, operatingCashFlow, capitalExpenditure (negative), netCashUsedForInvestingActivites,
                        dividendsPaid, netChangeInCash and, when the source provides them, the optional columns
                        ``buybacks``, ``debtProceeds``, ``debtRepayment`` (see ``OPTIONAL_CASHFLOW_COLUMNS``)

Everything returned is in INR crore (1 crore = 1e7 rupees). Flows are signed: money out is negative.
Items the source does not provide are ``None`` and listed in ``facts["unavailable"]``; investing cash flow is
never presented as capex.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

CRORE = 1e7
OPTIONAL_CASHFLOW_COLUMNS = ("buybacks", "debtProceeds", "debtRepayment")
FLOW_KEYS = ("cfo", "capex", "other_investing", "fcf", "dividends", "buybacks", "debt_net", "net_change")
CONVERSION_NORM_GOOD = 0.8      # general norm, not company-specific
CONVERSION_NORM_WEAK = 0.5
COUNTERCYCLICAL_CORR = -0.4


def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _cr(x) -> Optional[float]:
    v = _f(x)
    return None if v is None else v / CRORE


def _by_year(df: Optional[pd.DataFrame]) -> dict[int, pd.Series]:
    if df is None or getattr(df, "empty", True) or "year" not in df.columns:
        return {}
    return {int(r["year"]): r for _, r in df.iterrows()}


def _get(row: Optional[pd.Series], col: str) -> Optional[float]:
    if row is None or col not in row.index:
        return None
    return _f(row[col])


def _words(text: str, limit: int = 25) -> str:
    parts = text.split()
    return text if len(parts) <= limit else " ".join(parts[:limit]).rstrip(",;:") + "."


def verdict(v: str, yardstick: str, so_what: str, source_note: str) -> dict:
    assert v in ("good", "neutral", "concern")
    assert yardstick in ("own_history", "peers", "general_norm")
    return {"verdict": v, "yardstick": yardstick, "so_what": _words(so_what), "source_note": source_note}


# --------------------------------------------------------------------------------------
# TTM cash flow from half-year statements
# --------------------------------------------------------------------------------------
def derive_ttm_cashflow(annual_df: Optional[pd.DataFrame], half_df: Optional[pd.DataFrame]) -> Optional[dict]:
    """TTM = latest fiscal year + latest half-year - same half of the prior year (absolute units).

    Needs normalised cash-flow frames with a ``date`` column. Returns None unless the half-years line up
    (latest half ends about six months after the fiscal year end, prior half about one year earlier).
    """
    if annual_df is None or half_df is None or annual_df.empty or half_df.empty:
        return None
    if "date" not in annual_df.columns or "date" not in half_df.columns:
        return None
    fy = annual_df.sort_values("date", ascending=False).iloc[0]
    halves = half_df.sort_values("date", ascending=False)
    now = halves.iloc[0]
    gap = (pd.Timestamp(now["date"]) - pd.Timestamp(fy["date"])).days
    if not 150 <= gap <= 200:
        return None
    prior = [r for _, r in halves.iterrows() if 350 <= (pd.Timestamp(now["date"]) - pd.Timestamp(r["date"])).days <= 380]
    if not prior:
        return None
    prior = prior[0]
    cols = ["operatingCashFlow", "capitalExpenditure", "netCashUsedForInvestingActivites", "dividendsPaid",
            "netChangeInCash", *OPTIONAL_CASHFLOW_COLUMNS]
    out = {"period_end": pd.Timestamp(now["date"]).strftime("%Y-%m-%d")}
    for c in cols:
        a, b, p = _get(fy, c), _get(now, c), _get(prior, c)
        if None not in (a, b, p):
            out[c] = a + b - p
    return out if "operatingCashFlow" in out else None


# --------------------------------------------------------------------------------------
# facts
# --------------------------------------------------------------------------------------
def _row_flows(cf: Optional[pd.Series], investing_total: Optional[float] = None) -> dict:
    cfo = _get(cf, "operatingCashFlow")
    if cfo is None:
        cfo = _get(cf, "netCashProvidedByOperatingActivities")
    capex = _get(cf, "capitalExpenditure")
    capex = -abs(capex) if capex not in (None, 0.0) else None        # 0 is the normaliser's "missing" marker
    inv = _get(cf, "netCashUsedForInvestingActivites") if investing_total is None else investing_total
    div = _get(cf, "dividendsPaid")
    bb = _get(cf, "buybacks")
    proceeds, repay = _get(cf, "debtProceeds"), _get(cf, "debtRepayment")
    debt_net = (proceeds - abs(repay)) if proceeds is not None and repay is not None else None
    return {
        "cfo": cfo, "capex": capex,
        "other_investing": (inv - capex) if inv is not None and capex is not None else None,
        "fcf": (cfo + capex) if cfo is not None and capex is not None else None,
        "dividends": -abs(div) if div is not None else None,
        "buybacks": -abs(bb) if bb is not None else None,
        "debt_net": debt_net, "debt_repaid_gross": -abs(repay) if repay is not None else None,
        "net_change": _get(cf, "netChangeInCash"),
    }


def _to_cr(d: dict) -> dict:
    return {k: _cr(v) for k, v in d.items()}


def compute_cashflow_facts(income_df: Optional[pd.DataFrame], balance_df: Optional[pd.DataFrame],
                           cash_df: Optional[pd.DataFrame], ttm: Optional[dict] = None,
                           ttm_cashflow: Optional[dict] = None, years: int = 5) -> dict:
    """All cash-flow facts in crore, chronological (oldest to newest), plus diagnostics and three verdicts.

    ``ttm`` is the dict from ``get_finedge_ttm_and_latest_balance()["ttm"]`` (needs ``ebitda``); ``ttm_cashflow`` is
    an absolute-rupee dict as returned by ``derive_ttm_cashflow``. Both are optional.
    """
    inc, bal, cf = _by_year(income_df), _by_year(balance_df), _by_year(cash_df)
    unavailable: list[str] = []
    yrs = sorted(set(cf) & set(inc))[-years:]
    if not yrs:
        return {"available": False, "years": [], "unavailable": ["cash flow statement"], "per_year": {},
                "answers": {}, "cyclicality": None, "aggregate": {}, "uses_of_cash": {}, "working_capital_days": {}}

    per_year: dict[str, dict] = {}
    for y in yrs:
        flows = _to_cr(_row_flows(cf[y]))
        ebitda = _cr(_get(inc[y], "ebitda"))
        flows["ebitda"] = ebitda
        flows["conversion"] = (flows["cfo"] / ebitda) if flows["cfo"] is not None and ebitda and ebitda > 0 else None
        per_year[f"{y}A"] = flows

    # revenue growth (uses the year before the window too)
    growth: dict[str, Optional[float]] = {}
    for y in yrs:
        rev, prev = _get(inc[y], "revenue"), _get(inc.get(y - 1), "revenue")
        growth[f"{y}A"] = (rev / prev - 1) if rev and prev else None
    for lab, g in growth.items():
        per_year[lab]["revenue_growth"] = g

    # TTM column
    ttm_label = None
    if ttm_cashflow and ttm and _f(ttm.get("ebitda")):
        t = _to_cr(_row_flows(pd.Series(ttm_cashflow)))
        t["ebitda"] = _cr(ttm["ebitda"])
        t["conversion"] = (t["cfo"] / t["ebitda"]) if t["cfo"] is not None and t["ebitda"] and t["ebitda"] > 0 else None
        t["revenue_growth"] = None
        ttm_label = "TTM"
        per_year[ttm_label] = t
    elif ttm_cashflow is None:
        unavailable.append("TTM cash flow (needs half-year cash flow statements)")

    # availability flags
    labels = [f"{y}A" for y in yrs]
    names = {"capex": "capex (purchase of fixed and intangible assets)", "dividends": "dividends paid",
             "buybacks": "share buybacks", "debt_net": "net debt raised or repaid (borrowings and repayments)",
             "other_investing": "other investing flows", "net_change": "net change in cash"}
    for k, nm in names.items():
        if all(per_year[l][k] is None for l in labels):
            unavailable.append(f"{nm}: not provided by the data source")
    capex_available = any(per_year[l]["capex"] is not None for l in labels)
    capex_basis = ("capex = purchase of fixed and intangible assets as reported; treasury investments, grants and "
                   "other investing flows are shown separately" if capex_available else
                   "capex is not available from the data source; investing cash flow is NOT shown as capex")

    # aggregates
    def _sum(key: str) -> Optional[float]:
        vals = [per_year[l][key] for l in labels if per_year[l][key] is not None]
        return float(sum(vals)) if vals else None

    both = [l for l in labels if per_year[l]["cfo"] is not None and per_year[l]["ebitda"] and per_year[l]["ebitda"] > 0]
    agg_conv = (sum(per_year[l]["cfo"] for l in both) / sum(per_year[l]["ebitda"] for l in both)) if both else None
    aggregate = {"cfo": _sum("cfo"), "ebitda": _sum("ebitda"), "capex": _sum("capex"), "fcf": _sum("fcf"),
                 "conversion": agg_conv, "years": labels}

    uses = {"capex": _sum("capex"), "debt_repaid_gross": _sum("debt_repaid_gross"), "debt_net": _sum("debt_net"),
            "dividends": _sum("dividends"), "buybacks": _sum("buybacks"), "cash_retained": _sum("net_change"),
            "years": labels}

    # working capital days
    wc: dict[str, dict] = {}
    for y in yrs:
        rev, cogs = _get(inc[y], "revenue"), _get(inc[y], "costOfRevenue")
        b = bal.get(y)
        inv, rec, pay = _get(b, "inventory"), _get(b, "netReceivables"), _get(b, "accountPayables")
        if rev and cogs and None not in (inv, rec, pay):
            dio, dso, dpo = inv / cogs * 365, rec / rev * 365, pay / cogs * 365
            wc[f"{y}A"] = {"inventory_days": dio, "receivable_days": dso, "payable_days": dpo, "cash_conversion_days": dio + dso - dpo}
    if not wc:
        unavailable.append("working-capital days (inventory, receivables, payables)")

    # cyclicality
    pairs = [(growth[l], per_year[l]["conversion"], l) for l in labels
             if growth[l] is not None and per_year[l]["conversion"] is not None]
    cyc = None
    if len(pairs) >= 3:
        g = np.array([p[0] for p in pairs]); c = np.array([p[1] for p in pairs])
        corr = float(np.corrcoef(g, c)[0, 1]) if g.std() > 0 and c.std() > 0 else None
        down = [p[1] for p in pairs if p[0] < 0]
        up = [p[1] for p in pairs if p[0] >= 0]
        flag = bool(corr is not None and corr <= COUNTERCYCLICAL_CORR and down and up and np.mean(down) > np.mean(up))
        stmt = ("Cash conversion rises when revenue falls: cash is released from working capital in downturns and "
                "absorbed in recoveries, so a strong cash year in a downturn is partly one-off.") if flag else (
            "Cash conversion does not move against revenue, so cash flow is not mainly a working-capital effect." if corr is not None
            else "Not enough variation to test whether cash conversion depends on the revenue cycle.")
        cyc = {"correlation": corr, "n_years": len(pairs), "flag": flag,
               "avg_conversion_when_revenue_fell": float(np.mean(down)) if down else None,
               "avg_conversion_when_revenue_grew": float(np.mean(up)) if up else None,
               "years_revenue_fell": [p[2] for p in pairs if p[0] < 0], "statement": stmt}
    else:
        unavailable.append("cyclicality test (needs at least three years of revenue growth and cash conversion)")

    facts = {"available": True, "unit": "INR crore; money out is negative", "years": labels, "ttm_label": ttm_label,
             "per_year": per_year, "aggregate": aggregate, "uses_of_cash": uses, "working_capital_days": wc,
             "cyclicality": cyc, "capex_basis": capex_basis, "capex_available": capex_available,
             "unavailable": unavailable}
    facts["answers"] = cashflow_answers(facts)
    return facts


# --------------------------------------------------------------------------------------
# the three verdicts
# --------------------------------------------------------------------------------------
def cashflow_answers(facts: dict) -> dict:
    """Does profit turn into cash? Does the business fund itself? Is cash flattered by one-offs?"""
    labels, py = facts["years"], facts["per_year"]
    cyc, agg = facts.get("cyclicality"), facts["aggregate"]
    out = {}

    # 1. profit -> cash
    conv = agg.get("conversion")
    note = "general norm, not company-specific (about 80% or more of EBITDA reaching operating cash flow is healthy)"
    if conv is None:
        out["profit_to_cash"] = None
    else:
        flag = bool(cyc and cyc["flag"])
        if conv >= CONVERSION_NORM_GOOD and not flag:
            out["profit_to_cash"] = verdict("good", "general_norm", "Reported profit has been backed by cash over the period, so earnings quality looks sound.", note)
        elif conv >= CONVERSION_NORM_GOOD:
            out["profit_to_cash"] = verdict("neutral", "general_norm", "Cash backing for profit is strong overall, but it rises when sales fall, so it is not steady.", note)
        elif conv >= CONVERSION_NORM_WEAK:
            out["profit_to_cash"] = verdict("neutral", "general_norm", "Only part of reported profit reaches the bank, and the share swings widely from year to year.", note)
        else:
            out["profit_to_cash"] = verdict("concern", "general_norm", "Little reported profit turns into cash, so earnings deserve caution until cash catches up.", note)

    # 2. funds itself (five-year totals against shareholder payouts)
    fcf = agg.get("fcf")
    pos = [l for l in labels if py[l]["fcf"] is not None and py[l]["fcf"] > 0]
    known = [l for l in labels if py[l]["fcf"] is not None]
    payouts = sum(abs(v) for v in (facts["uses_of_cash"].get("dividends"), facts["uses_of_cash"].get("buybacks")) if v is not None)
    note2 = "report data (own five-year totals)"
    if fcf is None or not known:
        out["funds_itself"] = None
    elif fcf <= 0:
        out["funds_itself"] = verdict("concern", "own_history", "Free cash flow has not covered investment needs over five years, so the business leans on borrowing or reserves.", note2)
    elif fcf >= payouts and len(pos) >= 3:
        out["funds_itself"] = verdict("good", "own_history", "Free cash flow has covered shareholder payouts in most years, so the business has paid its own way.", note2)
    else:
        out["funds_itself"] = verdict("neutral", "own_history", "Free cash flow is positive in some years but does not reliably cover payouts and debt service.", note2)

    # 3. flattered by one-offs
    latest = labels[-1]
    cur_conv = py[latest]["conversion"]
    g_latest = py[latest].get("revenue_growth")
    wcd = facts["working_capital_days"]
    wc_release = (len(labels) >= 2 and labels[-2] in wcd and latest in wcd and
                  wcd[latest]["cash_conversion_days"] < wcd[labels[-2]]["cash_conversion_days"] - 10)
    note3 = "report data (own five-year history)"
    if cyc is None:
        out["flattered_by_one_offs"] = None
    elif cyc["flag"] and g_latest is not None and g_latest < 0 and cur_conv is not None and cur_conv > 1.0:
        out["flattered_by_one_offs"] = verdict("concern", "own_history", "The latest cash year is partly one-off: working capital was released in a downturn and may be absorbed again in recovery.", note3)
    elif cyc["flag"] or wc_release:
        out["flattered_by_one_offs"] = verdict("neutral", "own_history", "Cash moves against sales, so some strong cash years reflect working-capital timing, not lasting strength.", note3)
    else:
        out["flattered_by_one_offs"] = verdict("good", "own_history", "Cash conversion does not depend on sales swings, so recent cash looks repeatable.", note3)
    return out


# --------------------------------------------------------------------------------------
# render-ready output
# --------------------------------------------------------------------------------------
def _n(v: Optional[float], dec: int = 0) -> str:
    if v is None:
        return "n/a"
    s = f"{abs(v):,.{dec}f}"
    return f"({s})" if v < 0 else s


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:,.0f}%"


def cashflow_table_rows(facts: dict) -> tuple[list[str], list[list[str]]]:
    """(header, rows) in rupee crore, chronological; outflows in brackets; 'n/a' where the source has no figure."""
    if not facts.get("available"):
        return ["Rs crore"], []
    cols = list(facts["years"]) + ([facts["ttm_label"]] if facts.get("ttm_label") else [])
    py, wc = facts["per_year"], facts["working_capital_days"]
    spec = [("Operating cash flow (CFO)", "cfo"), ("Capex (fixed and intangible assets)", "capex"),
            ("Free cash flow (CFO less capex)", "fcf"), ("Other investing flows", "other_investing"),
            ("Dividends paid", "dividends"), ("Share buybacks", "buybacks"),
            ("Net debt raised / (repaid)", "debt_net"), ("Net change in cash", "net_change"), ("EBITDA", "ebitda")]
    rows = [[lab] + [_n(py[c].get(k)) for c in cols] for lab, k in spec]
    rows.append(["Cash conversion (CFO / EBITDA)"] + [_pct(py[c].get("conversion")) for c in cols])
    for lab, k in (("Inventory days", "inventory_days"), ("Receivable days", "receivable_days"), ("Payable days", "payable_days")):
        rows.append([lab] + [(f"{wc[c][k]:,.0f}" if c in wc else "n/a") for c in cols])
    return ["Rs crore"] + cols, rows


def cashflow_text_lines(facts: dict) -> list[str]:
    """Plain-text lines for the text layer and for text extraction of the PDF."""
    if not facts.get("available"):
        return ["CASH FLOW: not available. " + "; ".join(facts.get("unavailable", []))]
    py, agg, uses, cyc = facts["per_year"], facts["aggregate"], facts["uses_of_cash"], facts.get("cyclicality")
    L = ["CASH FLOW (INR crore; money out in brackets):"]
    for l in facts["years"] + ([facts["ttm_label"]] if facts.get("ttm_label") else []):
        r = py[l]
        L.append(f"  {l}: CFO {_n(r['cfo'])}, capex {_n(r['capex'])}, FCF {_n(r['fcf'])}, other investing {_n(r['other_investing'])}, "
                 f"dividends {_n(r['dividends'])}, buybacks {_n(r['buybacks'])}, net debt {_n(r['debt_net'])}, "
                 f"change in cash {_n(r['net_change'])}, EBITDA {_n(r['ebitda'])}, cash conversion {_pct(r['conversion'])}")
    L.append(f"  Five-year cash conversion (total CFO / total EBITDA): {_pct(agg.get('conversion'))}")
    L.append(f"  Five-year uses of cash: capex {_n(uses['capex'])}, gross debt repaid {_n(uses['debt_repaid_gross'])}, "
             f"dividends {_n(uses['dividends'])}, buybacks {_n(uses['buybacks'])}, cash retained {_n(uses['cash_retained'])}")
    if facts["working_capital_days"]:
        L.append("  Working-capital days (inventory / receivable / payable / cycle): " + "; ".join(
            f"{k} {v['inventory_days']:.0f}/{v['receivable_days']:.0f}/{v['payable_days']:.0f}/{v['cash_conversion_days']:.0f}"
            for k, v in facts["working_capital_days"].items()))
    if cyc:
        L.append("  Cyclicality: " + cyc["statement"])
    L.append("  Capex basis: " + facts["capex_basis"])
    q = {"profit_to_cash": "Does profit turn into cash?", "funds_itself": "Does the business fund itself?",
         "flattered_by_one_offs": "Is cash flattered by one-offs?"}
    for k, title in q.items():
        a = facts["answers"].get(k)
        if a:
            L.append(f"  {title} [{a['verdict']}, vs {a['yardstick'].replace('_', ' ')}] {a['so_what']} ({a['source_note']})")
    if facts["unavailable"]:
        L.append("  NOT AVAILABLE: " + "; ".join(facts["unavailable"]))
    return L
