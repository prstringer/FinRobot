#!/usr/bin/env python
# coding: utf-8
"""
Canonical, code-computed facts for one equity report.

Why this module exists
----------------------
Every report section (tagline, valuation, news, ...) used to receive the same raw-rupee DataFrames and leave the
unit conversion, the price basis of each multiple, net debt, EV, the peer median and the model-vs-peer EBITDA
distinction to the language model. Those are arithmetic and bookkeeping jobs, so they are done here, once:

* one unit (INR crore) is applied in ``scale_table_to_crore`` / ``build_report_context`` and reused by every section,
* every multiple carries the price, date, shares, net debt and EPS basis it was computed on,
* the model forecast, the peer-dataset trend projection and the FY-average-price peer multiples are stored under
  distinct, labelled keys,
* ``run_checks`` verifies the identities (price x shares = market cap, EPS x shares = PAT, P/E = price / EPS, ...),
* ``assert_section_contexts_identical`` proves every section prompt carries the same facts and table,
* ``write_reconciliation_log`` records the inputs behind each multiple for a human reviewer.

All rupee values in the context are in crore (1 crore = 1e7 rupees); shares are in millions; per-share values in rupees.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

CRORE = 1e7
FACTS_START = "=== CANONICAL FACTS (computed in code) ==="
FACTS_END = "=== END CANONICAL FACTS ==="
TABLE_START = "<<FINANCIAL TABLE>>"
TABLE_END = "<<END FINANCIAL TABLE>>"

# Money rows of the analysis table (absolute rupees in the CSV) that are rescaled to crore for text sections
MONEY_ROWS = ("Revenue", "Cost of Operations", "SG&A", "Contribution Profit", "EBITDA", "Employee Cost", "Other Operating Expenses")


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _f(x: Any) -> Optional[float]:
    """float or None (NaN/inf/None/non-numeric -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _cr(x: Any) -> Optional[float]:
    v = _f(x)
    return None if v is None else v / CRORE


def _row(df: pd.DataFrame, metric: str, col: str):
    """Value of ``metric`` in column ``col`` (the P/E row is matched by prefix)."""
    if df is None or df.empty or col not in df.columns:
        return None
    m = df["metrics"].astype(str)
    sel = df[(m == metric) | m.str.startswith(metric + " (")]
    return None if sel.empty else sel[col].iloc[0]


def _pct(x: Any) -> Optional[float]:
    """'8.4%' or 0.084-style strings/numbers -> percent number (8.4)."""
    if isinstance(x, str):
        x = x.replace("%", "").strip()
        return _f(x)
    return _f(x)


def fmt_cr(v: Optional[float], dec: int = 0) -> str:
    return "n/a" if v is None else f"{v:,.{dec}f} Cr"


def fmt_x(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:,.1f}x"


def _year_cols(df: pd.DataFrame, suffix: str) -> list[str]:
    return sorted([c for c in df.columns if isinstance(c, str) and re.fullmatch(rf"\d{{4}}{suffix}", c)])


def median_of(values: list[float]) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    return None if not vals else float(np.median(vals))


# --------------------------------------------------------------------------------------
# tables for the text sections (single canonical unit)
# --------------------------------------------------------------------------------------
def scale_table_to_crore(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the analysis table with money rows in INR crore (1 decimal). Ratios, EPS and % rows are untouched."""
    out = df.copy().astype("object")
    for metric in MONEY_ROWS:
        mask = out["metrics"] == metric
        for col in out.columns:
            if col == "metrics":
                continue
            for idx in out.index[mask]:
                v = _f(out.at[idx, col])
                out.at[idx, col] = None if v is None else round(v / CRORE, 1)
        out.loc[mask, "metrics"] = f"{metric} (INR crore)"
    return out


def peer_ebitda_to_crore(df: Optional[pd.DataFrame], years: Optional[list] = None) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    sel = df if years is None else df.loc[[y for y in years if y in df.index]]
    return (sel / CRORE).round(1)


# --------------------------------------------------------------------------------------
# context builder
# --------------------------------------------------------------------------------------
def build_report_context(*, ticker: str, company_name: str, as_of: str, final_df: pd.DataFrame,
                         financial_data: dict, company_metrics: dict, ttm_balance: Optional[dict] = None,
                         peer_ev_df: Optional[pd.DataFrame] = None,
                         peer_ebitda_actual: Optional[pd.DataFrame] = None,
                         peer_ebitda_projected: Optional[pd.DataFrame] = None,
                         peer_current_df: Optional[pd.DataFrame] = None,
                         technicals: Optional[dict] = None, fair_value_multiples=(12, 14, 16, 18),
                         extras: Optional[dict] = None) -> dict:
    """Assemble every figure the sections and the report need, with the basis of each stated."""
    ttm_balance = ttm_balance or {}
    cm = company_metrics or {}
    unavailable: list[str] = []

    income = financial_data.get("income_statement")
    balance = financial_data.get("balance_sheet")
    cash = financial_data.get("cash_flow")
    km = financial_data.get("key_metrics")

    actual_cols = _year_cols(final_df, "A")
    fc_cols = _year_cols(final_df, "E")
    latest_a = actual_cols[-1] if actual_cols else None
    latest_year = int(latest_a[:4]) if latest_a else None

    # ---- market ---------------------------------------------------------------------
    price = _f(cm.get("share_price"))
    mcap_cr = _f(cm.get("market_cap"))
    shares_mn = (mcap_cr * CRORE / price / 1e6) if price and mcap_cr else None
    if shares_mn is None:
        unavailable.append("shares outstanding (needs current price and market cap)")
    wk = cm.get("52w_range")
    market = {
        "price": price, "market_cap_cr": mcap_cr, "shares_mn": shares_mn,
        "shares_basis": "market cap / price (FinEdge quote)",
        "price_date": str(cm.get("price_time") or "")[:10] or None,
        "week52_low": _f(wk[0]) if isinstance(wk, (tuple, list)) else None,
        "week52_high": _f(wk[1]) if isinstance(wk, (tuple, list)) else None,
        "as_of": as_of,
    }

    # ---- latest balance sheet / net debt --------------------------------------------
    lb = ttm_balance.get("latest_balance") or {}
    if not lb and balance is not None and not balance.empty:
        b0 = balance.sort_values("date", ascending=False).iloc[0]
        lb = {"date": pd.Timestamp(b0["date"]).strftime("%Y-%m-%d"), "total_debt": b0.get("totalDebt"),
              "cash_and_st_investments": b0.get("cashAndShortTermInvestments"), "net_debt": b0.get("netDebt"),
              "equity": b0.get("totalStockholdersEquity"), "minority_interest": b0.get("minorityInterest")}
    bal = {
        "date": lb.get("date"), "total_debt_cr": _cr(lb.get("total_debt")),
        "cash_cr": _cr(lb.get("cash_and_st_investments")), "net_debt_cr": _cr(lb.get("net_debt")),
        "equity_cr": _cr(lb.get("equity")), "minority_interest_cr": _cr(lb.get("minority_interest")),
        "definition": "net debt = borrowings - (cash + other bank balances + current investments); minority interest not added to EV",
    }
    if bal["net_debt_cr"] is None:
        unavailable.append("latest net debt")

    ev_cr = (mcap_cr + bal["net_debt_cr"]) if mcap_cr is not None and bal["net_debt_cr"] is not None else None

    # ---- TTM ------------------------------------------------------------------------
    t = ttm_balance.get("ttm")
    ttm = None
    if t:
        ttm = {"period_end": t["period_end"], "quarters": t["quarters"], "revenue_cr": _cr(t["revenue"]),
               "ebitda_cr": _cr(t["ebitda"]), "depreciation_cr": _cr(t["depreciation"]),
               "interest_cr": _cr(t["interest"]), "pretax_cr": _cr(t["pretax_income"]),
               "pat_cr": _cr(t["net_income"]), "eps": round(_f(t["eps"]), 2) if _f(t["eps"]) is not None else None, "shares_mn": (_f(t["shares"]) or 0) / 1e6 or None}
    else:
        unavailable.append("TTM P&L (four consecutive quarters not available)")

    # ---- latest fiscal year (actual) ------------------------------------------------
    fy = None
    if latest_a:
        fy = {"label": latest_a, "revenue_cr": _cr(_row(final_df, "Revenue", latest_a)),
              "ebitda_cr": _cr(_row(final_df, "EBITDA", latest_a)), "eps": _f(_row(final_df, "EPS", latest_a))}
    inc_latest = None
    if income is not None and not income.empty and latest_year is not None:
        sel = income[income["year"] == latest_year]
        inc_latest = None if sel.empty else sel.iloc[0]
    fy_pat_cr = _cr(inc_latest["netIncome"]) if inc_latest is not None else None
    fy_shares_mn = (_f(inc_latest["weightedAverageShsOutDil"]) or 0) / 1e6 if inc_latest is not None else None
    fy_equity_cr = None
    if balance is not None and not balance.empty and latest_year is not None:
        bsel = balance[balance["year"] == latest_year]
        fy_equity_cr = None if bsel.empty else _cr(bsel.iloc[0]["totalStockholdersEquity"])

    # ---- current valuation (at report date) -----------------------------------------
    cur = {
        "as_of": as_of, "price": price, "market_cap_cr": mcap_cr, "net_debt_cr": bal["net_debt_cr"],
        "net_debt_date": bal["date"], "ev_cr": ev_cr, "ev_definition": "market cap + latest net debt",
        "ebitda_latest_fy_cr": fy["ebitda_cr"] if fy else None, "ebitda_latest_fy_label": latest_a,
        "ebitda_ttm_cr": ttm["ebitda_cr"] if ttm else None, "ttm_period_end": ttm["period_end"] if ttm else None,
    }
    cur["ev_ebitda_latest_fy"] = (ev_cr / cur["ebitda_latest_fy_cr"]) if ev_cr and cur["ebitda_latest_fy_cr"] else None
    cur["ev_ebitda_ttm"] = (ev_cr / cur["ebitda_ttm_cr"]) if ev_cr and cur["ebitda_ttm_cr"] else None
    cur["eps_latest_fy"] = fy["eps"] if fy else None
    cur["eps_ttm"] = ttm["eps"] if ttm else None
    cur["pe_latest_fy_at_current_price"] = (price / cur["eps_latest_fy"]) if price and cur["eps_latest_fy"] else None
    cur["pe_ttm_computed"] = (price / cur["eps_ttm"]) if price and cur["eps_ttm"] else None
    cur["pe_ttm_api"] = _f(cm.get("fwd_pe"))
    cur["pe_ttm_basis"] = cm.get("pe_basis") or "price / TTM EPS"
    cur["pb"] = _f(cm.get("pb_ratio"))
    cur["ev_ebitda_forecast"] = {}

    # ---- ratios on one stated period definition -------------------------------------
    roe_ttm_calc = (ttm["pat_cr"] / bal["equity_cr"] * 100) if ttm and ttm["pat_cr"] is not None and bal["equity_cr"] else None
    roe_fy_calc = (fy_pat_cr / fy_equity_cr * 100) if fy_pat_cr is not None and fy_equity_cr else None
    ratios = {
        "period_definition": "TTM = four quarters to " + (ttm["period_end"] if ttm else "n/a"),
        "roe_ttm_pct": _f(cm.get("roe")), "roe_ttm_pct_recomputed": roe_ttm_calc,
        "roe_ttm_basis": cm.get("roe_basis") or "TTM net income / latest closing equity",
        "roe_latest_fy_pct": roe_fy_calc, "roe_latest_fy_basis": "FY net income to owners / FY closing equity",
        "pe_ttm": cur["pe_ttm_api"], "pe_ttm_basis": cur["pe_ttm_basis"],
        "dividend_yield_pct": _f(cm.get("dividend_yield")),
        "dividend_basis": cm.get("dividend_basis") or "dividends in the last 365 days / current price",
        "net_margin_ttm_pct": (ttm["pat_cr"] / ttm["revenue_cr"] * 100) if ttm and ttm["revenue_cr"] else None,
        "net_margin_latest_fy_pct": (fy_pat_cr / fy["revenue_cr"] * 100) if fy and fy["revenue_cr"] and fy_pat_cr is not None else None,
    }

    # ---- historical FY multiples (fiscal-year AVERAGE price) ------------------------
    fy_series = {}
    if km is not None and not km.empty:
        for _, r in km.sort_values("year").iterrows():
            fy_series[f"{int(r['year'])}A"] = {
                "avg_price": _f(r.get("averagePrice")), "pe": _f(r.get("peRatio")), "eps": _f(r.get("netIncomePerShare")),
                "market_cap_cr": _cr(r.get("marketCap")), "net_debt_cr": _cr(r.get("netDebt")),
                "ev_cr": _cr(r.get("enterpriseValue")), "ev_ebitda": _f(r.get("enterpriseValueOverEBITDA")),
                "shares_mn": (_f(r.get("shares")) or 0) / 1e6 or None,
            }
    fy_series_basis = "fiscal-year AVERAGE price x diluted weighted shares; FY-end net debt (not the current price)"

    # ---- model forecast (our projection) --------------------------------------------
    bridge = final_df.attrs.get("forecast_bridge")
    assumptions = final_df.attrs.get("forecast_bridge_assumptions")
    model_forecast = {}
    for c in fc_cols:
        entry = {"revenue_cr": _cr(_row(final_df, "Revenue", c)), "ebitda_cr": _cr(_row(final_df, "EBITDA", c)),
                 "ebitda_margin_pct": _pct(_row(final_df, "EBITDA Margin", c)),
                 "eps": _f(_row(final_df, "EPS", c)), "pe_at_current_price": _f(_row(final_df, "PE Ratio", c))}
        if bridge is not None and c in bridge.index:
            b = bridge.loc[c]
            entry.update({"depreciation_cr": _cr(b["depreciation"]), "ebit_cr": _cr(b["ebit"]),
                          "interest_cr": _cr(b["interest"]), "other_income_cr": _cr(b["other_income"]),
                          "pretax_cr": _cr(b["pretax_income"]), "tax_cr": _cr(b["tax"]),
                          "pat_owners_cr": _cr(b["pat_owners"]), "shares_mn": b["shares"] / 1e6})
        model_forecast[c] = entry
    if ev_cr:
        for _y, _e in model_forecast.items():
            if _e.get("ebitda_cr"):
                cur["ev_ebitda_forecast"][_y] = ev_cr / _e["ebitda_cr"]
    if not model_forecast:
        unavailable.append("model forecast")
    if bridge is None:
        unavailable.append("forecast P&L bridge (D&A / interest / tax inputs)")

    # ---- peers ----------------------------------------------------------------------
    subject = ticker.upper()
    peer = {"ev_ebitda": None, "trend_projection": None, "ebitda_actual_cr": None}
    if peer_ev_df is not None and not peer_ev_df.empty:
        ev = peer_ev_df.copy()
        if "year" in ev.columns:
            ev = ev.set_index("year")
        yr = ev.index.max()
        vals = {c: _f(ev.loc[yr, c]) for c in ev.columns}
        peers_only = [v for c, v in vals.items() if c.upper() != subject]
        everyone = [v for v in vals.values()]
        ranked = sorted(((v, c) for c, v in vals.items() if v is not None and c.upper() != subject))
        med = median_of(peers_only)
        # which peer sits at the median (odd count) for narrative attribution; None when it is an average of two
        med_name = next((c for v, c in ranked if med is not None and abs(v - med) < 1e-9), None)
        peer["ev_ebitda"] = {
            "year": f"{int(yr)}A", "values": vals, "subject": vals.get(subject) if subject in vals else
            next((v for c, v in vals.items() if c.upper() == subject), None),
            "median_excl_subject": med, "median_incl_subject": median_of(everyone),
            "median_peer_name": med_name, "n_peers": len([v for v in peers_only if v is not None]),
            "policy": "peer median EXCLUDES the subject company; median incl. subject is shown only for reference",
            "basis": "fiscal-year AVERAGE price for every company in the table (not current prices)",
        }
    peer["current"] = None
    if peer_current_df is not None and not peer_current_df.empty:
        cdf = peer_current_df
        rows = {t: {k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v))) for k, v in r.items()}
                for t, r in cdf.iterrows()}
        def _med(col, include):
            return median_of([_f(r[col]) for t, r in rows.items() if include or t.upper() != subject])
        peer["current"] = {
            "as_of": as_of, "rows": rows,
            "median_excl_subject_fy": _med("ev_ebitda_fy", False), "median_incl_subject_fy": _med("ev_ebitda_fy", True),
            "median_excl_subject_ttm": _med("ev_ebitda_ttm", False), "median_incl_subject_ttm": _med("ev_ebitda_ttm", True),
            "basis": "current price (latest quote) x shares + latest net debt, over latest-FY EBITDA and over TTM EBITDA",
            "policy": "medians EXCLUDE the subject company",
        }
    if peer_ebitda_actual is not None and not peer_ebitda_actual.empty:
        peer["ebitda_actual_cr"] = {int(y): {c: _cr(v) for c, v in row.items()} for y, row in peer_ebitda_actual.iterrows()}
    if peer_ebitda_projected is not None and peer_ebitda_actual is not None and not peer_ebitda_projected.empty:
        new_years = [y for y in peer_ebitda_projected.index if y not in peer_ebitda_actual.index]
        if new_years:
            y = new_years[0]
            peer["trend_projection"] = {
                "year": int(y), "ebitda_cr": {c: _cr(v) for c, v in peer_ebitda_projected.loc[y].items()},
                "method": "each company's EBITDA extrapolated one year at its own mean historical YoY growth",
                "is_consensus": False,
                "subject_gap_cr": ((model_forecast.get(fc_cols[0], {}).get("ebitda_cr") - _cr(peer_ebitda_projected.loc[y].get(ticker.upper())))
                                   if fc_cols and model_forecast.get(fc_cols[0], {}).get("ebitda_cr") is not None
                                   and _cr(peer_ebitda_projected.loc[y].get(ticker.upper())) is not None else None),
                "note": "mechanical extrapolation, NOT analyst consensus; no analyst consensus is available from FinEdge",
            }

    # ---- fair-value grid vs CURRENT EV ----------------------------------------------
    grid = []
    first_fc = fc_cols[0] if fc_cols else None
    base_ebitda = model_forecast.get(first_fc, {}).get("ebitda_cr") if first_fc else None
    if base_ebitda and ev_cr and bal["net_debt_cr"] is not None and shares_mn:
        for m in fair_value_multiples:
            ev_m = m * base_ebitda
            eq = ev_m - bal["net_debt_cr"]
            px = eq * CRORE / (shares_mn * 1e6)
            grid.append({"multiple": m, "ev_cr": ev_m, "vs_current_ev_pct": (ev_m / ev_cr - 1) * 100,
                         "implied_price": px, "vs_current_price_pct": (px / price - 1) * 100 if price else None})
    fair_value = {"ebitda_year": first_fc, "ebitda_cr": base_ebitda, "current_ev_cr": ev_cr, "grid": grid,
                  "note": "implied EV = multiple x model FY EBITDA; compared with CURRENT EV, not the FY-end EV"}

    # ---- derived operating fields (computed, not narrated) ---------------------------
    derived = {"unavailable": unavailable}
    derived["shares_outstanding_mn"] = shares_mn
    if cash is not None and not cash.empty:
        derived["fcf_cr"] = {f"{int(r['year'])}A": _cr(r.get("freeCashFlow")) for _, r in cash.sort_values("year").iterrows()}
        derived["capex_cr"] = {f"{int(r['year'])}A": _cr(-abs(r["capitalExpenditure"])) for _, r in cash.sort_values("year").iterrows()}
        derived["cfo_cr"] = {f"{int(r['year'])}A": _cr(r.get("operatingCashFlow")) for _, r in cash.sort_values("year").iterrows()}
    else:
        unavailable.append("cash flow (FCF, capex)")
    if income is not None and not income.empty:
        icov = {}
        for _, r in income.sort_values("year").iterrows():
            ebit, intr = _f(r.get("operatingIncome")), _f(r.get("interestExpense"))
            icov[f"{int(r['year'])}A"] = (ebit / intr) if ebit is not None and intr else None
        derived["interest_coverage_ebit_x"] = icov
    if income is not None and balance is not None and not income.empty and not balance.empty:
        wc = {}
        inc_i, bal_i = income.set_index("year"), balance.set_index("year")
        for y in sorted(set(inc_i.index) & set(bal_i.index)):
            rev, cogs = _f(inc_i.loc[y, "revenue"]), _f(inc_i.loc[y, "costOfRevenue"])
            inv, rec, pay = _f(bal_i.loc[y].get("inventory")), _f(bal_i.loc[y].get("netReceivables")), _f(bal_i.loc[y].get("accountPayables"))
            if rev and cogs and None not in (inv, rec, pay):
                dio, dso, dpo = inv / cogs * 365, rec / rev * 365, pay / cogs * 365
                wc[f"{int(y)}A"] = {"inventory_days": dio, "receivable_days": dso, "payable_days": dpo, "cash_conversion_days": dio + dso - dpo}
        derived["working_capital_days"] = wc
        if not wc:
            unavailable.append("working-capital days (inventory / receivables / payables)")

    rev_row = {c: _f(_row(final_df, "Revenue", c)) for c in actual_cols}
    rev_row = {c: v for c, v in rev_row.items() if v}
    if len(rev_row) >= 2:
        ys = sorted(rev_row)
        n = int(ys[-1][:4]) - int(ys[0][:4])
        derived["revenue_cagr_pct"] = ((rev_row[ys[-1]] / rev_row[ys[0]]) ** (1 / n) - 1) * 100 if n > 0 else None
        derived["revenue_growth_pct"] = [(rev_row[b] / rev_row[a] - 1) * 100 for a, b in zip(ys[:-1], ys[1:])]

    # ---- technicals -----------------------------------------------------------------
    tech = dict(technicals or {})
    if tech.get("price") is not None:
        p = tech["price"]
        for k in ("sma20", "sma50", "sma200"):
            if tech.get(k):
                tech[f"price_vs_{k}_pct"] = (p / tech[k] - 1) * 100
    tech["week52_low"] = market["week52_low"] if market["week52_low"] is not None else tech.get("low_52w")
    tech["week52_high"] = market["week52_high"] if market["week52_high"] is not None else tech.get("high_52w")
    if tech.get("price") is not None and tech.get("week52_high"):
        tech["pct_below_52w_high"] = (tech["price"] / tech["week52_high"] - 1) * 100

    ctx = {
        "meta": {"ticker": ticker, "company_name": company_name, "as_of": as_of, "currency": "INR",
                 "unit": "INR crore (1 crore = 10,000,000 rupees); shares in millions; per-share in rupees",
                 "latest_actual_year": latest_a, "fiscal_year_end": "31 March"},
        "market": market, "balance_latest": bal, "current_valuation": cur, "ttm": ttm, "ratios": ratios,
        "latest_fy": {**(fy or {}), "pat_cr": fy_pat_cr, "shares_mn": fy_shares_mn, "closing_equity_cr": fy_equity_cr},
        "fy_series": fy_series, "fy_series_basis": fy_series_basis,
        "model_forecast": model_forecast, "model_forecast_assumptions": assumptions,
        "model_forecast_note": "our own driver-based projection; it is NOT analyst consensus",
        "peers": peer, "fair_value": fair_value, "derived": derived, "technicals": tech,
        # cost taxonomy (industrial vs generic labels) and the forecast margin-cap flag, both set by the analysis step
        "taxonomy": final_df.attrs.get("taxonomy"), "forecast_caps": final_df.attrs.get("margin_cap"),
    }
    from . import insights as _ins
    from . import cost_taxonomy as _ct
    try:
        ctx["insights"] = _ct.apply_text(_ins.compute_insights(ctx, final_df, financial_data, extras), ctx.get("taxonomy"))
    except Exception as e:   # explanatory facts must never block the report
        print(f"Warning: could not compute explanatory facts and verdicts: {e}")
        ctx["insights"] = {}
    ctx["extras"] = extras or {}
    ctx["checks"] = run_checks(ctx)
    ctx["prompt_block"] = _ct.apply_text(render_prompt_block(ctx), ctx.get("taxonomy"))
    return ctx


# --------------------------------------------------------------------------------------
# consistency checks + reconciliation log
# --------------------------------------------------------------------------------------
def _check(name: str, expected: Optional[float], actual: Optional[float], tol: float, mode: str = "rel",
           detail: str = "") -> dict:
    if expected is None or actual is None:
        return {"name": name, "expected": expected, "actual": actual, "tolerance": tol, "ok": None,
                "detail": (detail + " ").strip() + "(skipped: input unavailable)"}
    diff = abs(expected - actual) / (abs(expected) if mode == "rel" and expected else 1.0)
    return {"name": name, "expected": expected, "actual": actual, "tolerance": tol, "ok": bool(diff <= tol), "detail": detail}


def run_checks(ctx: dict) -> list[dict]:
    """Identities that must hold. ``ok`` is None when an input was unavailable (never a silent pass)."""
    m, cur, ttm, fy = ctx["market"], ctx["current_valuation"], ctx["ttm"], ctx["latest_fy"]
    out = []
    if m["price"] and m["shares_mn"] and m["market_cap_cr"]:
        out.append(_check("price x shares = market cap (Cr)", m["market_cap_cr"],
                          m["price"] * m["shares_mn"] * 1e6 / CRORE, 0.015))
    out.append(_check("EV = market cap + net debt (Cr)", cur["ev_cr"],
                      (m["market_cap_cr"] + cur["net_debt_cr"]) if m["market_cap_cr"] is not None and cur["net_debt_cr"] is not None else None, 1e-6))
    if fy.get("pat_cr") is not None and fy.get("eps") is not None and fy.get("shares_mn"):
        out.append(_check(f"EPS x shares = PAT ({fy.get('label')})", fy["pat_cr"], fy["eps"] * fy["shares_mn"] * 1e6 / CRORE, 0.01))
    if ttm and ttm.get("shares_mn") and ttm.get("eps") is not None and ttm.get("pat_cr") is not None:
        out.append(_check("EPS x shares = PAT (TTM)", ttm["pat_cr"], ttm["eps"] * ttm["shares_mn"] * 1e6 / CRORE, 0.03))
    out.append(_check("P/E TTM = price / TTM EPS (API vs computed)", cur["pe_ttm_api"], cur["pe_ttm_computed"], 0.02))
    if cur.get("eps_latest_fy") and cur.get("price"):
        out.append(_check("P/E latest FY at current price = price / EPS", cur["pe_latest_fy_at_current_price"], cur["price"] / cur["eps_latest_fy"], 1e-6))
    # each historical FY multiple vs its own price basis
    for label, r in ctx["fy_series"].items():
        if r["avg_price"] and r["eps"] and r["pe"]:
            out.append(_check(f"P/E {label} = FY-avg price / EPS", r["pe"], r["avg_price"] / r["eps"], 0.01, detail=f"avg price {r['avg_price']}"))
        if r["ev_cr"] and r["ev_ebitda"] and label == ctx["meta"]["latest_actual_year"] and ctx["latest_fy"].get("ebitda_cr"):
            out.append(_check(f"EV/EBITDA {label} = FY EV / FY EBITDA", r["ev_ebitda"], r["ev_cr"] / ctx["latest_fy"]["ebitda_cr"], 0.01))
    ratios = ctx["ratios"]
    out.append(_check("ROE TTM: API vs TTM PAT / closing equity (pp)", ratios["roe_ttm_pct"], ratios["roe_ttm_pct_recomputed"], 0.5, mode="abs",
                      detail="tolerance in percentage points"))
    # forecast bridge reconciliation
    for y, e in ctx["model_forecast"].items():
        if "pretax_cr" in e:
            pbt = e["ebitda_cr"] - e["depreciation_cr"] - e["interest_cr"] + e["other_income_cr"]
            out.append(_check(f"forecast {y}: EBITDA - D&A - interest + other income = PBT (Cr)", e["pretax_cr"], pbt, 1e-6))
            if e.get("shares_mn") and e.get("eps") is not None:
                out.append(_check(f"forecast {y}: EPS x shares = PAT (Cr)", e["pat_owners_cr"], e["eps"] * e["shares_mn"] * 1e6 / CRORE, 0.01))
            if e.get("pe_at_current_price") and e.get("eps") and m["price"]:
                out.append(_check(f"forecast {y}: P/E = current price / EPS", e["pe_at_current_price"], m["price"] / e["eps"], 0.01))
    peers = ctx["peers"]["ev_ebitda"]
    if peers:
        vals = [v for c, v in peers["values"].items() if v is not None and c.upper() != ctx["meta"]["ticker"].upper()]
        out.append(_check("peer median (excl. subject) recomputed", peers["median_excl_subject"], float(np.median(vals)) if vals else None, 1e-9))
    pc = ctx["peers"].get("current")
    if pc:
        for col, key in (("ev_ebitda_fy", "median_excl_subject_fy"), ("ev_ebitda_ttm", "median_excl_subject_ttm")):
            vals = [r[col] for t, r in pc["rows"].items() if r.get(col) is not None and t.upper() != ctx["meta"]["ticker"].upper()]
            out.append(_check(f"current-price peer median ({col}) recomputed", pc[key], float(np.median(vals)) if vals else None, 1e-9))
        for t, r in pc["rows"].items():
            if r.get("price") and r.get("market_cap_cr") and r.get("net_debt_cr") is not None and r.get("ev_cr") is not None:
                out.append(_check(f"{t}: current EV = market cap + net debt (Cr)", r["ev_cr"], r["market_cap_cr"] + r["net_debt_cr"], 1e-6))
        subj = pc["rows"].get(ctx["meta"]["ticker"].upper()) or pc["rows"].get(ctx["meta"]["ticker"])
        if subj and ctx["current_valuation"].get("ev_ebitda_latest_fy") and subj.get("ev_ebitda_fy"):
            out.append(_check("subject current EV/EBITDA: peer table = current-valuation block", ctx["current_valuation"]["ev_ebitda_latest_fy"], subj["ev_ebitda_fy"], 0.01))
    return out


def failed_checks(ctx: dict) -> list[dict]:
    return [c for c in ctx.get("checks", []) if c["ok"] is False]


def write_reconciliation_log(ctx: dict, path: str, extra_notes: Optional[list[str]] = None) -> None:
    """Short markdown log: the price, date, shares, net debt and EPS basis behind every multiple."""
    m, cur, ttm, bal = ctx["market"], ctx["current_valuation"], ctx["ttm"], ctx["balance_latest"]
    L = [f"# Reconciliation log: {ctx['meta']['company_name']} ({ctx['meta']['ticker']})", "",
         f"Report date: {ctx['meta']['as_of']}. Unit: {ctx['meta']['unit']}.", "",
         "## Inputs behind each multiple", "",
         "| Multiple | Value | Price (basis) | Shares (mn) | Net debt (Cr, date) | EPS / EBITDA basis |", "|---|---|---|---|---|---|"]
    nd_now = f"{bal['net_debt_cr']:,.1f} ({bal['date']})" if bal["net_debt_cr"] is not None else "n/a"
    sh = f"{m['shares_mn']:,.2f}" if m["shares_mn"] else "n/a"
    L.append(f"| EV/EBITDA, latest FY, CURRENT | {fmt_x(cur['ev_ebitda_latest_fy'])} | Rs {m['price']} (current, {m['as_of']}) | {sh} | {nd_now} | {ctx['meta']['latest_actual_year']} EBITDA {fmt_cr(cur['ebitda_latest_fy_cr'])} |")
    L.append(f"| EV/EBITDA, TTM, CURRENT | {fmt_x(cur['ev_ebitda_ttm'])} | Rs {m['price']} (current) | {sh} | {nd_now} | TTM EBITDA {fmt_cr(cur['ebitda_ttm_cr'])} to {cur['ttm_period_end']} |")
    L.append(f"| P/E, TTM | {fmt_x(cur['pe_ttm_api'])} | Rs {m['price']} (current) | n/a | n/a | TTM EPS Rs {cur['eps_ttm'] if cur['eps_ttm'] is not None else 'n/a'} |")
    L.append(f"| P/E, latest FY at current price | {fmt_x(cur['pe_latest_fy_at_current_price'])} | Rs {m['price']} (current) | n/a | n/a | {ctx['meta']['latest_actual_year']} EPS Rs {cur['eps_latest_fy']} |")
    for label, r in ctx["fy_series"].items():
        L.append(f"| P/E {label} (historical) | {fmt_x(r['pe'])} | Rs {r['avg_price']} (FY AVERAGE) | {r['shares_mn']:,.1f} | {r['net_debt_cr']:,.1f} (FY end) | EPS Rs {r['eps']} |" if r['shares_mn'] and r['net_debt_cr'] is not None else f"| P/E {label} | {fmt_x(r['pe'])} | Rs {r['avg_price']} (FY AVERAGE) | n/a | n/a | EPS Rs {r['eps']} |")
        L.append(f"| EV/EBITDA {label} (historical) | {fmt_x(r['ev_ebitda'])} | Rs {r['avg_price']} (FY AVERAGE) | same | same | EV {fmt_cr(r['ev_cr'])} |")
    for y, e in ctx["model_forecast"].items():
        L.append(f"| P/E {y} (forecast) | {fmt_x(e.get('pe_at_current_price'))} | Rs {m['price']} (current) | {e.get('shares_mn', float('nan')):,.1f} | n/a | model EPS Rs {e.get('eps')} (from EBITDA bridge) |")
    L += ["", "## Ratio period definitions", "",
          f"* ROE (TTM): {ctx['ratios']['roe_ttm_pct']}% = {ctx['ratios']['roe_ttm_basis']}; recomputed {ctx['ratios']['roe_ttm_pct_recomputed']}%",
          f"* ROE ({ctx['meta']['latest_actual_year']}): {ctx['ratios']['roe_latest_fy_pct']}% = {ctx['ratios']['roe_latest_fy_basis']}",
          f"* P/E (TTM): {ctx['ratios']['pe_ttm']}x = {ctx['ratios']['pe_ttm_basis']}",
          f"* Dividend yield: {ctx['ratios']['dividend_yield_pct']}% = {ctx['ratios']['dividend_basis']}",
          f"* {ctx['ratios']['period_definition']}",
          "", "## Peer median", ""]
    pe = ctx["peers"]["ev_ebitda"]
    if pe:
        L += [f"* {pe['year']} EV/EBITDA, {pe['n_peers']} peers: median excl. subject {fmt_x(pe['median_excl_subject'])}"
              f"{' (' + pe['median_peer_name'] + ')' if pe['median_peer_name'] else ''}; incl. subject {fmt_x(pe['median_incl_subject'])}",
              f"* Policy: {pe['policy']}. Basis: {pe['basis']}."]
    L += ["", "## Checks", "", "| Check | Expected | Actual | Result |", "|---|---|---|---|"]
    for c in ctx["checks"]:
        res = "PASS" if c["ok"] else ("FAIL" if c["ok"] is False else "SKIPPED")
        ex = "n/a" if c["expected"] is None else f"{c['expected']:,.4g}"
        ac = "n/a" if c["actual"] is None else f"{c['actual']:,.4g}"
        L.append(f"| {c['name']} | {ex} | {ac} | {res} |")
    if ctx["derived"]["unavailable"] or extra_notes:
        L += ["", "## Unavailable inputs / notes", ""]
        L += [f"* {u}" for u in ctx["derived"]["unavailable"]] + [f"* {n}" for n in (extra_notes or [])]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------
# prompt plumbing
# --------------------------------------------------------------------------------------
def render_prompt_block(ctx: dict) -> str:
    """Plain-text block of canonical facts, identical for every section."""
    m, cur, bal, ttm = ctx["market"], ctx["current_valuation"], ctx["balance_latest"], ctx["ttm"]
    r, mf, fv = ctx["ratios"], ctx["model_forecast"], ctx["fair_value"]
    L = [FACTS_START,
         f"Unit: {ctx['meta']['unit']}. Report date {ctx['meta']['as_of']}. Fiscal year ends 31 March (2026A = FY2025-26).",
         "Use these figures exactly as written; do not convert, rescale or recompute them.",
         "",
         f"CURRENT MARKET: price Rs {m['price']} (close {m.get('price_date') or m['as_of']}) | shares {m['shares_mn']:,.2f} mn | market cap {fmt_cr(m['market_cap_cr'])}"
         if m["price"] and m["shares_mn"] else "CURRENT MARKET: unavailable",
         f"LATEST BALANCE SHEET ({bal['date']}): total debt {fmt_cr(bal['total_debt_cr'])}; cash and investments {fmt_cr(bal['cash_cr'])}; "
         f"net debt {fmt_cr(bal['net_debt_cr'], 1)}" + (" (net cash)" if bal["net_debt_cr"] is not None and bal["net_debt_cr"] < 0 else ""),
         f"CURRENT EV (market cap + latest net debt): {fmt_cr(cur['ev_cr'])}",
         f"CURRENT EV/EBITDA: {fmt_x(cur['ev_ebitda_latest_fy'])} on {cur['ebitda_latest_fy_label']} EBITDA {fmt_cr(cur['ebitda_latest_fy_cr'])}; "
         f"{fmt_x(cur['ev_ebitda_ttm'])} on TTM EBITDA {fmt_cr(cur['ebitda_ttm_cr'])} (TTM to {cur['ttm_period_end']})",
         f"CURRENT P/E: {fmt_x(cur['pe_ttm_api'])} on TTM EPS Rs {cur['eps_ttm']}; {fmt_x(cur['pe_latest_fy_at_current_price'])} on "
         f"{cur['ebitda_latest_fy_label']} EPS Rs {cur['eps_latest_fy']} (both at the current price)",
         f"PERIOD BASIS: ROE TTM {r['roe_ttm_pct']:.2f}% (TTM net income / closing equity); ROE {ctx['meta']['latest_actual_year']} "
         f"{r['roe_latest_fy_pct']:.2f}% (FY net income / FY closing equity); net margin TTM {r['net_margin_ttm_pct']:.1f}% vs FY {r['net_margin_latest_fy_pct']:.1f}%; "
         f"dividend yield {r['dividend_yield_pct']}% (last 365 days / current price)"
         if None not in (r["roe_ttm_pct"], r["roe_latest_fy_pct"], r["net_margin_ttm_pct"], r["net_margin_latest_fy_pct"]) else
         f"PERIOD BASIS: {r['period_definition']}",
         "",
         f"HISTORICAL MULTIPLES (price basis = {ctx['fy_series_basis']}; they are NOT at today's price):"]
    for label, x in ctx["fy_series"].items():
        L.append(f"  {label}: avg price Rs {x['avg_price']}, P/E {fmt_x(x['pe'])} (EPS Rs {x['eps']}), EV {fmt_cr(x['ev_cr'])}, EV/EBITDA {fmt_x(x['ev_ebitda'])}")
    L += ["", "MODEL FORECAST (our own projection, NOT analyst consensus; P/E uses the current price):"]
    for y, e in mf.items():
        L.append(f"  {y}: revenue {fmt_cr(e['revenue_cr'])}, EBITDA {fmt_cr(e['ebitda_cr'])} ({e['ebitda_margin_pct']}% margin), "
                 f"EPS Rs {e['eps']}, P/E {fmt_x(e['pe_at_current_price'])}")
    a = ctx.get("model_forecast_assumptions")
    if a:
        L.append(f"  EPS bridge: EBITDA - D&A ({a['da_pct_of_revenue']*100:.1f}% of revenue) - interest ({fmt_cr(a['interest_expense_flat']/CRORE)} flat) "
                 f"+ other income ({fmt_cr(a['other_income_flat']/CRORE)} flat) - tax ({a['tax_rate']*100:.1f}%) - minorities, over {a['diluted_shares']/1e6:,.1f} mn shares")
    pe = ctx["peers"]["ev_ebitda"]
    if pe:
        vals = ", ".join(f"{c} {v:.1f}x" for c, v in sorted(pe["values"].items(), key=lambda kv: (kv[1] is None, kv[1])) if v is not None)
        L += ["", f"PEER EV/EBITDA {pe['year']} ({pe['basis']}): {vals}",
              f"  Peer median EXCLUDING {ctx['meta']['ticker']}: {fmt_x(pe['median_excl_subject'])}"
              f"{' (' + pe['median_peer_name'] + ')' if pe['median_peer_name'] else ''}; including it: {fmt_x(pe['median_incl_subject'])}. Use the excluding figure."]
    pc = ctx["peers"].get("current")
    if pc:
        vals = ", ".join(f"{t} {fmt_x(r.get('ev_ebitda_fy'))} FY / {fmt_x(r.get('ev_ebitda_ttm'))} TTM"
                         for t, r in sorted(pc["rows"].items(), key=lambda kv: (kv[1].get("ev_ebitda_fy") is None, kv[1].get("ev_ebitda_fy") or 0)))
        L += ["", f"PEER EV/EBITDA AT CURRENT PRICES ({pc['as_of']}; {pc['basis']}): {vals}",
              f"  Medians excluding {ctx['meta']['ticker']}: {fmt_x(pc['median_excl_subject_fy'])} on latest-FY EBITDA, {fmt_x(pc['median_excl_subject_ttm'])} on TTM EBITDA. "
              "Compare the subject's CURRENT multiples with these, and its fiscal-year-average multiples with the fiscal-year-average peer figures above; do not cross the two bases."]
    tp = ctx["peers"]["trend_projection"]
    if tp:
        L += ["", f"PEER-DATASET TREND PROJECTION FY{tp['year']} EBITDA (mechanical extrapolation, NOT consensus; separate from the model forecast): "
              + ", ".join(f"{c} {fmt_cr(v)}" for c, v in tp["ebitda_cr"].items())]
        if tp.get("subject_gap_cr") is not None:
            L.append(f"  Gap: our model FY{tp['year']} EBITDA is {fmt_cr(tp['subject_gap_cr'])} {'above' if tp['subject_gap_cr'] >= 0 else 'below'} the peer-dataset projection for {ctx['meta']['ticker']}.")
    if fv["grid"]:
        L += ["", f"FAIR-VALUE GRID on model {fv['ebitda_year']} EBITDA {fmt_cr(fv['ebitda_cr'])}; current EV {fmt_cr(fv['current_ev_cr'])}:"]
        for g in fv["grid"]:
            L.append(f"  {g['multiple']}x -> EV {fmt_cr(g['ev_cr'])} ({g['vs_current_ev_pct']:+.0f}% vs current EV); implied price Rs {g['implied_price']:.0f} ({g['vs_current_price_pct']:+.0f}% vs Rs {m['price']})")
    t = ctx["technicals"]
    if t.get("rsi14") is not None and not (ctx.get("extras") or {}).get("technical"):
        L += ["", f"TECHNICALS (close Rs {t.get('price')}): RSI14 {t.get('rsi14')} ({t.get('rsi_signal')}); MACD {t.get('macd')} / signal {t.get('macd_signal')} / histogram {t.get('macd_histogram')}; "
              f"SMA20 {t.get('sma20')}, SMA50 {t.get('sma50')}, SMA200 {t.get('sma200')}; 52-week range Rs {t.get('week52_low')} - Rs {t.get('week52_high')}"]
    d = ctx["derived"]
    if d.get("fcf_cr"):
        L += ["", "CASH FLOW (Cr): " + "; ".join(f"{k} CFO {fmt_cr(d['cfo_cr'].get(k))}, capex {fmt_cr(d['capex_cr'].get(k))}, FCF {fmt_cr(v)}" for k, v in d["fcf_cr"].items())]
    if d.get("interest_coverage_ebit_x"):
        L.append("INTEREST COVERAGE (EBIT / interest): " + ", ".join(f"{k} {fmt_x(v)}" for k, v in d["interest_coverage_ebit_x"].items()))
    if d.get("working_capital_days"):
        L.append("CASH CONVERSION DAYS: " + ", ".join(f"{k} {v['cash_conversion_days']:.0f}" for k, v in d["working_capital_days"].items()))
    tx, caps = ctx.get("taxonomy"), ctx.get("forecast_caps")
    if tx and tx.get("kind") == "industrial":
        L.append("COST TAXONOMY: industrial. Costs are materials (consumed + purchases + change in inventory), employee cost and other operating "
                 "expenses; power and fuel are inside other operating expenses (the provider does not split them). "
                 "EBITDA = revenue from operations less total operating expenses (Ind-AS), before finance costs, depreciation and amortisation.")
    if caps and caps.get("message"):
        L.append("FORECAST MARGIN CAP: " + caps["message"])
    if ctx.get("insights"):
        from . import insights as _ins
        L.append(_ins.render_insights_block(ctx["insights"]))
    for blk in (ctx.get("extras") or {}).get("fact_blocks", []):
        L.append(blk)
    if d["unavailable"]:
        L.append("NOT AVAILABLE: " + "; ".join(d["unavailable"]))
    L.append(FACTS_END)
    return "\n".join(L)


def current_valuation_rows(ctx: dict) -> list[tuple[str, str, str]]:
    """(metric, value, basis) rows for the 'Current valuation (at report date)' table, shared by HTML and PDF."""
    m, cur, bal = ctx["market"], ctx["current_valuation"], ctx["balance_latest"]
    fy = ctx["meta"]["latest_actual_year"]
    hist = ctx["fy_series"].get(fy, {})
    rows = [
        ("Share price", f"Rs {m['price']:,.2f}" if m["price"] else "n/a",
         f"close on {m['price_date'] or m['as_of']} (price snapshot); report date {m['as_of']}"),
        ("Shares outstanding", f"{m['shares_mn']:,.1f} mn" if m["shares_mn"] else "n/a", m["shares_basis"]),
        ("Market cap", fmt_cr(m["market_cap_cr"]), "exchange quote"),
        ("Net debt", fmt_cr(bal["net_debt_cr"], 1) + (" (net cash)" if bal["net_debt_cr"] is not None and bal["net_debt_cr"] < 0 else ""),
         f"latest balance sheet, {bal['date']}"),
        ("Enterprise value", fmt_cr(cur["ev_cr"]), "market cap + latest net debt"),
        (f"EV/EBITDA ({fy})", fmt_x(cur["ev_ebitda_latest_fy"]), f"current EV / {fy} EBITDA {fmt_cr(cur['ebitda_latest_fy_cr'])}"),
        ("EV/EBITDA (TTM)", fmt_x(cur["ev_ebitda_ttm"]), f"current EV / TTM EBITDA {fmt_cr(cur['ebitda_ttm_cr'])} to {cur['ttm_period_end']}"),
        ("P/E (TTM)", fmt_x(cur["pe_ttm_api"]), f"current price / TTM EPS Rs {cur['eps_ttm']}"),
        (f"P/E ({fy}, at current price)", fmt_x(cur["pe_latest_fy_at_current_price"]), f"current price / {fy} EPS Rs {cur['eps_latest_fy']}"),
    ]
    for y, v in (cur.get("ev_ebitda_forecast") or {}).items():
        rows.append((f"EV/EBITDA ({y}, model)", fmt_x(v), f"current EV / model {y} EBITDA {fmt_cr(ctx['model_forecast'][y]['ebitda_cr'])}"))
    if hist.get("ev_ebitda") is not None:
        rows.append((f"Memo: {fy} EV/EBITDA, FY-avg price", fmt_x(hist["ev_ebitda"]),
                     f"historical series, FY-average price Rs {hist['avg_price']} and FY-end net debt; not comparable with current multiples"))
    return rows


def peer_multiples_table(ctx: dict) -> Optional[tuple[list[str], list[list[str]]]]:
    """Peer EV/EBITDA on both price bases: (header, rows) shared by the HTML and PDF reports.

    Columns: fiscal-year-average-price multiple (the historical series) next to the current-price snapshot.
    """
    pe, pc = ctx["peers"].get("ev_ebitda"), ctx["peers"].get("current")
    if not pe:
        return None
    subject = ctx["meta"]["ticker"].upper()
    fy = pe["year"]
    cur_fy = next((r["fy_label"] for r in (pc["rows"].values() if pc else []) if r.get("fy_label")), fy)
    header = ["Company", f"EV/EBITDA {fy}\n(FY-avg price)"]
    if pc:
        header += ["Price (Rs)\n(current)", f"EV/EBITDA {cur_fy}\n(current price)", "EV/EBITDA TTM\n(current price)"]
    rows = []
    order = sorted(pe["values"], key=lambda c: (c.upper() != subject, c))
    for t in order:
        name = f"{t} (subject)" if t.upper() == subject else t
        row = [name, fmt_x(pe["values"][t])]
        if pc:
            r = pc["rows"].get(t) or {}
            row += [f"{r['price']:,.2f}" if r.get("price") else "n/a", fmt_x(r.get("ev_ebitda_fy")), fmt_x(r.get("ev_ebitda_ttm"))]
        rows.append(row)
    med = ["Peer median (excl. subject)", fmt_x(pe["median_excl_subject"])]
    inc = ["Median incl. subject", fmt_x(pe["median_incl_subject"])]
    if pc:
        med += ["", fmt_x(pc["median_excl_subject_fy"]), fmt_x(pc["median_excl_subject_ttm"])]
        inc += ["", fmt_x(pc["median_incl_subject_fy"]), fmt_x(pc["median_incl_subject_ttm"])]
    return header, rows + [med, inc]


def technical_block(tf: dict) -> str:
    """Fact lines for the text layer from technical_analysis facts (readings, four tests, divergence, price vs earnings)."""
    from . import technical_analysis as ta
    L = ["TECHNICAL READINGS (computed):"]
    L += [f"  {a}: {b}" + (f" ({c})" if c else "") for a, b, c in ta.readings_table_rows(tf)]
    L.append("  Four independent tests: " + "; ".join(f"{t[0]} {t[2]} ({t[3]})" for t in ta.tests_rows(tf)))
    for k in ("verdict_sentence", "divergence_sentence"):
        if tf.get(k):
            L.append(f"  {tf[k]}")
    pve = tf.get("price_vs_earnings") or {}
    if pve.get("sentence"):
        L.append(f"  Price versus earnings: {pve['sentence']}")
    return "\n".join(L)


def render_table_block(table_md: str) -> str:
    return f"{TABLE_START}\n{table_md}\n{TABLE_END}"


def build_text_data(ctx: dict, final_df: pd.DataFrame, peer_ebitda_actual: Optional[pd.DataFrame],
                    peer_ev_df: Optional[pd.DataFrame], base: Optional[dict] = None) -> dict:
    """The ``data`` dict handed to every text section: crore-scaled tables plus the canonical facts block."""
    data = dict(base or {})
    from . import cost_taxonomy as _ct
    data["financial_metrics"] = _ct.rename_rows(scale_table_to_crore(final_df), ctx.get("taxonomy"))   # display names per taxonomy
    data["taxonomy"] = ctx.get("taxonomy")
    data["peer_ebitda"] = peer_ebitda_to_crore(peer_ebitda_actual)   # actual years only; projection lives in the facts block
    data["peer_ev_ebitda"] = peer_ev_df
    data["report_context"] = ctx
    data["facts_block"] = ctx["prompt_block"]
    notes = [n for n in data.get("data_notes", []) if "absolute rupees" not in n]
    notes.insert(0, "All rupee amounts in the tables and facts are in INR crore (1 crore = 10,000,000 rupees). Fiscal years end in March: 2026A is FY2025-26.")
    notes.append("Peer EV/EBITDA values are on each company's fiscal-year average price, not current prices.")
    for n in _ct.notes(ctx.get("taxonomy")) + [(ctx.get("forecast_caps") or {}).get("message")]:
        if n and n not in notes:
            notes.append(n)
    data["data_notes"] = notes
    return data


def assert_section_contexts_identical(prompts: dict[str, str]) -> None:
    """Every section prompt must carry byte-identical canonical facts and financial table (raises AssertionError)."""
    def grab(text: str, start: str, end: str) -> Optional[str]:
        i, j = text.find(start), text.find(end)
        return None if i < 0 or j < 0 else text[i:j + len(end)]
    for start, end, what in ((FACTS_START, FACTS_END, "canonical facts"), (TABLE_START, TABLE_END, "financial table")):
        found = {sec: grab(p, start, end) for sec, p in prompts.items()}
        missing = [s for s, v in found.items() if v is None]
        assert not missing, f"{what} missing from section prompt(s): {missing}"
        distinct = {v for v in found.values()}
        assert len(distinct) == 1, f"{what} differs between sections: {sorted(found)}"


def _canonical_crore_values(ctx: dict) -> dict[str, float]:
    """Every crore-denominated figure the sections were given, keyed by a readable name."""
    canon: dict[str, float] = {}

    def add(name, v):
        if v:
            canon[name] = abs(float(v))
    lf, cur, ttm = ctx["latest_fy"], ctx["current_valuation"], ctx["ttm"] or {}
    for k in ("revenue_cr", "ebitda_cr", "pat_cr"):
        add(f"{lf.get('label')} {k[:-3]}", lf.get(k))
    for y, e in ctx["model_forecast"].items():
        for k, v in e.items():
            if k.endswith("_cr"):
                add(f"{y} {k[:-3]}", v)
    for k in ("revenue_cr", "ebitda_cr", "pat_cr", "pretax_cr", "interest_cr", "depreciation_cr"):
        add(f"TTM {k[:-3]}", ttm.get(k))
    add("market cap", ctx["market"]["market_cap_cr"])
    add("current EV", cur["ev_cr"])
    for k in ("total_debt_cr", "cash_cr", "net_debt_cr", "equity_cr"):
        add(f"balance {k[:-3]}", ctx["balance_latest"].get(k))
    for label, r in ctx["fy_series"].items():
        add(f"{label} EV", r.get("ev_cr"))
        add(f"{label} market cap", r.get("market_cap_cr"))
        add(f"{label} net debt", r.get("net_debt_cr"))
    for g in ctx["fair_value"]["grid"]:
        add(f"{g['multiple']}x EV", g["ev_cr"])
    for y, row in (ctx["peers"].get("ebitda_actual_cr") or {}).items():
        for c, v in row.items():
            add(f"{c} {y} EBITDA", v)
    tp = ctx["peers"].get("trend_projection")
    if tp:
        for c, v in tp["ebitda_cr"].items():
            add(f"{c} projected EBITDA", v)
        add("model vs peer-dataset gap", tp.get("subject_gap_cr"))
    for k in ("fcf_cr", "capex_cr", "cfo_cr"):
        for y, v in (ctx["derived"].get(k) or {}).items():
            add(f"{y} {k[:-3]}", v)
    ins = ctx.get("insights") or {}
    ol = ins.get("operating_leverage") or {}
    add("operating leverage revenue change", ol.get("d_revenue_cr"))
    add("operating leverage EBITDA change", ol.get("d_ebitda_cr"))
    for r in (ins.get("ttm_comparison") or {}).get("rows", []):
        if r.get("unit") == "Cr":
            add(f"FY {r['label']}", r["fy"])
            add(f"TTM {r['label']}", r["ttm"])

    def walk(x, name):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{name}.{k}")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            add(name, x)
    walk((ctx.get("extras") or {}).get("cashflow"), "cashflow")
    return canon


def lint_narrative_units(text: str, ctx: dict) -> list[str]:
    """Flag crore amounts that match no canonical figure but are exactly 10x or one-tenth of one (a unit slip).

    A number that equals any canonical figure is accepted, so a correct EBITDA of 977 Cr is not mistaken for
    one-tenth of revenue. Warnings only: the narrative is never edited here.
    """
    canon = _canonical_crore_values(ctx)
    values = list(canon.values())
    warnings = []
    for m in re.finditer(r"([\d,]+(?:\.\d+)?)\s*(?:crore|Cr\b)", text or ""):
        n = _f(m.group(1).replace(",", ""))
        if not n:
            continue
        if any(abs(n - v) <= max(0.01 * v, 0.6) for v in values):
            continue
        for name, v in canon.items():
            if abs(n / v - 10) < 0.03 or abs(n / v - 0.1) < 0.003:
                warnings.append(f"'{m.group(0)}' is 10x / one-tenth of canonical {name} ({v:,.1f} Cr)")
                break
    return warnings


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def save_context(ctx: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(ctx), fh, indent=2, ensure_ascii=False)


def load_context(path: str, ticker: Optional[str] = None) -> Optional[dict]:
    """Context saved by the analysis step, or None (missing file, unreadable, or a different ticker)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            ctx = json.load(fh)
    except (OSError, ValueError):
        return None
    if ticker and str(ctx.get("meta", {}).get("ticker", "")).upper() != str(ticker).upper():
        return None
    return ctx


def format_report_date(ctx: Optional[dict] = None) -> str:
    """Exact report date, e.g. '15 November 2026': the analysis date of the context, else today."""
    import datetime as _dt
    d = None
    if ctx:
        try:
            d = _dt.datetime.strptime(str(ctx["meta"]["as_of"])[:10], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            d = None
    d = d or _dt.date.today()
    return f"{d.day} {d.strftime('%B %Y')}"
