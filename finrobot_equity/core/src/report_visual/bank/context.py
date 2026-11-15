#!/usr/bin/env python
# coding: utf-8
"""
Bank report context (``report_context.json`` for banks), written by the analysis step.

It carries the same skeleton keys as the non-financial context (meta, market, current_valuation, fy_series, peers, ...)
so existing consumers of ``report_context.json`` degrade gracefully (EBITDA, EV and net-debt fields are None for banks),
plus bank-specific blocks: ``bank_series`` (every metric of the bank analysis table), ``peers.bank`` (latest-year ratios,
P/B at fiscal-year-average prices and P/B at today's prices, with medians that EXCLUDE the subject) and ``extras.technical``.

Money is INR crore, shares in millions, ratios in percent numbers (3.12 means 3.12%).
"""

from __future__ import annotations

import datetime as _dt
import statistics
from typing import Any, Optional

import numpy as np
import pandas as pd

CRORE = 1e7


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(str(x).replace("%", "").replace(",", "").strip())
    except ValueError:
        return None
    return v if np.isfinite(v) else None


def _med(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


# ----------------------------------------------------------------------------------------------
# API helpers (FinEdge)
# ----------------------------------------------------------------------------------------------
def fetch_bank_ttm(client, symbol: str, statement_type: str = "s") -> Optional[dict]:
    """Trailing four quarters for a bank (net income, EPS, NII, total income) in absolute rupees, or None."""
    from modules.finedge_api import FinEdgeError, normalize_income_statement, normalize_symbol
    symbol = normalize_symbol(symbol)
    try:
        rows = client.financials(symbol, "pl", "quarterly", statement_type)
    except FinEdgeError:
        return None
    if not rows:
        return None
    q = normalize_income_statement(rows, "quarterly")
    if q.empty or "date" not in q:
        return None
    q = q.sort_values("date", ascending=False)
    q = q[q["date"].notna()].head(4).reset_index(drop=True)
    if len(q) < 4:
        return None
    gaps = [(q.loc[i, "date"] - q.loc[i + 1, "date"]).days for i in range(3)]
    if not all(80 <= g <= 100 for g in gaps):
        return None
    shares = _f(q.loc[0, "weightedAverageShsOutDil"])
    eps_q = []
    for e, n, s in zip(q["eps"], q["netIncome"], q["weightedAverageShsOutDil"]):
        e, n, s = _f(e), _f(n), _f(s)
        eps_q.append(e if e is not None else (n / s if n is not None and s else 0.0))
    return {"period_end": q.loc[0, "date"].strftime("%Y-%m-%d"), "quarters": [d.strftime("%Y-%m-%d") for d in q["date"]],
            "net_income": float(q["netIncome"].sum()), "eps": float(sum(eps_q)),
            "net_interest_income": float(q["netInterestIncome"].sum()) if "netInterestIncome" in q else None,
            "total_income": float(q["revenue"].sum()) if "revenue" in q else None, "shares": shares}


def fetch_latest_equity(client, symbol: str, statement_type: str = "s") -> Optional[dict]:
    """Latest shareholders' equity (annual, half-yearly or quarterly balance sheet, whichever is newest)."""
    from modules.finedge_api import FinEdgeError, normalize_balance_sheet, normalize_symbol
    symbol = normalize_symbol(symbol)
    best = None
    for period in ("annual", "halfyearly", "quarterly"):
        try:
            bs = normalize_balance_sheet(client.financials(symbol, "bs", period, statement_type), period)
        except FinEdgeError:
            continue
        if bs.empty or bs["date"].isna().all():
            continue
        row = bs.sort_values("date", ascending=False).iloc[0]
        if best is None or row["date"] > best["date"]:
            best = row
    if best is None or _f(best.get("totalStockholdersEquity")) is None:
        return None
    return {"date": best["date"].strftime("%Y-%m-%d"), "equity": float(best["totalStockholdersEquity"])}


def fetch_peer_current_pb(client, tickers: list[str], statement_type: str = "s") -> dict[str, dict]:
    """P/B at today's price for each bank: current market cap / latest equity. Skips symbols the plan does not cover."""
    from modules.finedge_api import FinEdgeError, _nonzero, normalize_symbol
    out: dict[str, dict] = {}
    for t in tickers:
        sym = normalize_symbol(t)
        try:
            q = client.quote(sym)
            price, mcap = _nonzero(q.get("current_price")), _nonzero(q.get("market_cap"))
            eq = fetch_latest_equity(client, sym, statement_type)
        except FinEdgeError:
            continue
        if not eq or not price or not mcap or pd.isna(price) or pd.isna(mcap):
            continue
        out[sym] = {"price": float(price), "market_cap_cr": float(mcap), "equity_cr": eq["equity"] / CRORE, "equity_date": eq["date"],
                    "pb_now": float(mcap) * CRORE / eq["equity"]}
    return out


# ----------------------------------------------------------------------------------------------
# context
# ----------------------------------------------------------------------------------------------
def bank_series(final_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """{metric: {'2026A': value}} for every numeric cell of the bank analysis table (actual and forecast columns)."""
    cols = [c for c in final_df.columns if isinstance(c, str) and len(c) == 5 and c[:4].isdigit() and c[4] in "AE"]
    out: dict[str, dict[str, float]] = {}
    for _, r in final_df.iterrows():
        vals = {c: _f(r[c]) for c in sorted(cols)}
        vals = {c: v for c, v in vals.items() if v is not None}
        if vals:
            out[str(r["metrics"])] = vals
    return out


def _peer_block(peer_bank_df: Optional[pd.DataFrame], peer_current: dict, subject: str) -> Optional[dict]:
    if peer_bank_df is None or peer_bank_df.empty:
        return None
    df = peer_bank_df.copy()
    year = int(df["year"].max())
    df = df[df["year"] == year]
    rows = {}
    for _, r in df.iterrows():
        t = str(r["ticker"]).upper()
        def pct100(k):
            v = _f(r.get(k))
            return None if v is None else round(v * 100, 4)
        rows[t] = {"pb_fy_avg": _f(r.get("P/B")), "roe": pct100("ROE"), "roa": pct100("ROA"), "nim": pct100("NIM"),
                   "gnpa": pct100("GNPA"), "nnpa": pct100("NNPA"), "cet1": pct100("CET1")}
        cur = (peer_current or {}).get(t)
        rows[t]["pb_now"] = cur["pb_now"] if cur else None
        rows[t]["price_now"] = cur["price"] if cur else None
    meds = {}
    for k in ("pb_fy_avg", "pb_now", "roe", "roa", "nim", "gnpa", "nnpa", "cet1"):
        meds[k] = _med([r[k] for t, r in rows.items() if t != subject])
    return {"year": year, "rows": rows, "median_excl_subject": meds, "n_peers": len([t for t in rows if t != subject]),
            "policy": "peer medians EXCLUDE the subject company",
            "basis": "latest fiscal year; pb_fy_avg uses fiscal-year-average prices, pb_now uses today's price over latest equity"}


def build_bank_context(*, ticker: str, company_name: str, as_of: str, final_df: pd.DataFrame, financial_data: dict,
                       company_metrics: dict, ttm: Optional[dict], latest_equity: Optional[dict],
                       peer_bank_df: Optional[pd.DataFrame], peer_current: Optional[dict],
                       technical_facts: Optional[dict], warnings: Optional[list] = None) -> dict:
    cm = company_metrics or {}
    series = bank_series(final_df)
    actual = sorted(c for c in (series.get("EPS") or {}) if c.endswith("A"))
    fc = sorted(c for c in (series.get("EPS") or {}) if c.endswith("E"))
    latest = actual[-1] if actual else None
    price, mcap = _f(cm.get("share_price")), _f(cm.get("market_cap"))
    shares_mn = (mcap * CRORE / price / 1e6) if price and mcap else None
    wk = cm.get("52w_range")
    eps_fy = (series.get("EPS") or {}).get(latest)
    eps_ttm = (ttm["eps"] if ttm else None)
    pe_api = _f(cm.get("fwd_pe"))
    equity_cr = latest_equity["equity"] / CRORE if latest_equity else None
    bvps_latest = (latest_equity["equity"] / (shares_mn * 1e6)) if latest_equity and shares_mn else None
    km = financial_data.get("key_metrics")
    fy_series: dict[str, dict] = {}
    if km is not None and not km.empty:
        for _, r in km.sort_values("year").iterrows():
            lab = f"{int(r['year'])}A"
            fy_series[lab] = {"avg_price": _f(r.get("averagePrice")), "pe": _f(r.get("peRatio")), "pb": _f(r.get("pbRatio")),
                              "eps": _f(r.get("netIncomePerShare")), "market_cap_cr": (_f(r.get("marketCap")) or 0) / CRORE or None,
                              "net_debt_cr": None, "ev_cr": None, "ev_ebitda": None, "shares_mn": (_f(r.get("shares")) or 0) / 1e6 or None,
                              "bvps": (series.get("Book Value per Share") or {}).get(lab)}
    inc = financial_data.get("income_statement")
    pat_cr = None
    if inc is not None and not inc.empty and latest:
        sel = inc[inc["year"] == int(latest[:4])]
        pat_cr = (_f(sel.iloc[0]["netIncome"]) or 0) / CRORE if not sel.empty else None
    forecast = {}
    for c in fc:
        e = (series.get("EPS") or {}).get(c)
        forecast[c] = {"revenue_cr": None, "ebitda_cr": None, "ebitda_margin_pct": None, "eps": e,
                       "pe_at_current_price": (price / e) if price and e else None}
    unavailable = ["enterprise value and net debt (not meaningful for a bank)",
                   "segment and branch data, LCR, deposit mix and branch counts (not provided by FinEdge)",
                   "analyst consensus forecasts (not provided by FinEdge)"]
    if not ttm:
        unavailable.append("TTM earnings (four consecutive quarters not found)")
    if not equity_cr:
        unavailable.append("latest book value (no recent balance sheet)")
    ctx = {
        "kind": "bank",
        "meta": {"ticker": ticker, "company_name": company_name, "as_of": as_of, "currency": "INR", "unit": "INR crore; shares in millions",
                 "latest_actual_year": latest, "fiscal_year_end": "31 March", "kind": "bank", "sector": cm.get("sector")},
        "market": {"price": price, "market_cap_cr": mcap, "shares_mn": shares_mn, "shares_basis": "market cap / price (FinEdge quote)",
                   "price_date": str(cm.get("price_time") or "")[:10] or None,
                   "week52_low": _f(wk[0]) if isinstance(wk, (tuple, list)) else None,
                   "week52_high": _f(wk[1]) if isinstance(wk, (tuple, list)) else None, "as_of": as_of},
        "balance_latest": {"date": latest_equity["date"] if latest_equity else None, "total_debt_cr": None, "cash_cr": None, "net_debt_cr": None,
                           "equity_cr": equity_cr, "minority_interest_cr": None, "bvps": bvps_latest,
                           "definition": "net debt is not meaningful for a bank; equity is shareholders' equity"},
        "current_valuation": {
            "as_of": as_of, "price": price, "market_cap_cr": mcap, "net_debt_cr": None, "net_debt_date": None, "ev_cr": None,
            "ev_definition": "not used for banks", "ebitda_latest_fy_cr": None, "ebitda_latest_fy_label": latest, "ebitda_ttm_cr": None,
            "ttm_period_end": ttm["period_end"] if ttm else None, "ev_ebitda_latest_fy": None, "ev_ebitda_ttm": None,
            "ev_ebitda_forecast": {}, "eps_latest_fy": eps_fy, "eps_ttm": eps_ttm,
            "pe_latest_fy_at_current_price": (price / eps_fy) if price and eps_fy else None,
            "pe_ttm_computed": (price / eps_ttm) if price and eps_ttm else None, "pe_ttm_api": pe_api,
            "pe_ttm_basis": cm.get("pe_basis") or "price / TTM EPS", "pb": _f(cm.get("pb_ratio")),
            "pb_now": (price / bvps_latest) if price and bvps_latest else None,
            "pb_basis": "snapshot price / book value per share (latest balance sheet)"},
        "ttm": ({"period_end": ttm["period_end"], "quarters": ttm["quarters"], "revenue_cr": (ttm["total_income"] or 0) / CRORE or None,
                 "ebitda_cr": None, "pat_cr": ttm["net_income"] / CRORE, "eps": round(ttm["eps"], 2),
                 "shares_mn": (ttm["shares"] or 0) / 1e6 or None} if ttm else None),
        "ratios": {"period_definition": "TTM = four quarters to " + (ttm["period_end"] if ttm else "n/a"),
                   "roe_ttm_pct": _f(cm.get("roe")), "roe_ttm_pct_recomputed": (ttm["net_income"] / CRORE / equity_cr * 100) if ttm and equity_cr else None,
                   "roe_ttm_basis": cm.get("roe_basis") or "TTM net income / latest closing equity", "roe_latest_fy_pct": (series.get("ROE") or {}).get(latest),
                   "roe_latest_fy_basis": "FY net income / FY closing equity (FinEdge ratio)", "pe_ttm": pe_api,
                   "pe_ttm_basis": cm.get("pe_basis") or "price / TTM EPS", "dividend_yield_pct": _f(cm.get("dividend_yield")),
                   "dividend_basis": cm.get("dividend_basis") or "dividends in the last 365 days / current price",
                   "net_margin_ttm_pct": None, "net_margin_latest_fy_pct": None},
        "latest_fy": {"label": latest, "revenue_cr": ((series.get("Total Income") or {}).get(latest) or 0) / CRORE or None, "ebitda_cr": None,
                      "eps": eps_fy, "pat_cr": pat_cr, "shares_mn": (fy_series.get(latest) or {}).get("shares_mn"),
                      "closing_equity_cr": None, "bvps": (series.get("Book Value per Share") or {}).get(latest)},
        "fy_series": fy_series, "fy_series_basis": "fiscal-year AVERAGE price (not the current price)",
        "model_forecast": forecast,
        "model_forecast_assumptions": {"note": "EPS forecast = last EPS grown at the revenue growth assumption; no P&L bridge for banks"},
        "model_forecast_note": "our own simple projection; it is NOT analyst consensus",
        "peers": {"ev_ebitda": None, "current": None, "ebitda_actual_cr": None, "trend_projection": None,
                  "bank": _peer_block(peer_bank_df, peer_current or {}, ticker.upper())},
        "fair_value": {"ebitda_year": None, "ebitda_cr": None, "current_ev_cr": None, "grid": [], "note": "bank fair-value cases are built from P/B and P/E"},
        "derived": {"unavailable": unavailable, "shares_outstanding_mn": shares_mn},
        "technicals": {}, "insights": {}, "checks": [], "prompt_block": "",
        "extras": {"fact_blocks": [], "technical": technical_facts or None},
        "bank_series": series,
    }
    if warnings:
        ctx["data_warnings"] = list(warnings)
    return ctx
