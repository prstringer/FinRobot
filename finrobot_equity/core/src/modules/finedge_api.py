#!/usr/bin/env python
# coding: utf-8
"""
FinEdge API adapter for Indian equities (NSE/BSE).

Returns data in the same shapes the FMP functions in ``market_data_api`` return,
so the rest of the report pipeline can work unchanged:

* ``get_finedge_comprehensive_financial_data`` -> dict of DataFrames using FMP
  column names (revenue, costOfRevenue, grossProfit, ebitda, totalDebt, ...).
* ``get_finedge_company_metrics`` -> same keys as
  ``get_comprehensive_company_metrics`` (share_price, market_cap, ...).
* ``get_finedge_price_history`` / ``get_finedge_technical_indicators``.
* ``get_finedge_company_news`` -> exchange announcements in FMP news format.
* ``combine_peer_financial_data_finedge`` -> peer EBITDA and EV/EBITDA.
* ``build_bank_metrics_table`` -> analysis table for banks (NII, NIM, GNPA...).

Units: FinEdge returns statement values in absolute rupees, and the adapter keeps
them that way (FMP returns absolute dollars). ``market_cap`` in the metrics dict
is in rupee crore, the "large unit" for INR used by ``modules.currency``.

Data notes (see finrobot_equity/docs/finedge_data_quality.md):
* FinEdge has no EBITDA, gross profit or SG&A fields. They are derived:
  EBITDA = revenue from operations - (total expenses - finance costs - D&A),
  i.e. operating EBITDA excluding other income and exceptional items.
  SG&A = employee benefit expense + other expenses.
* Zero is often used where data is missing. Ratios and multiples equal to 0
  are treated as missing.
* The free plan only allows the sample symbols; other symbols raise
  ``FinEdgePlanError`` (HTTP 401).
"""

from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

FINEDGE_BASE_URL = "https://data.finedgeapi.com"
TOKEN_ENV_VAR = "FINEDGE_API_TOKEN"
CRORE = 1e7
NIFTY50_SYMBOL = "NIF50"


class FinEdgeError(RuntimeError):
    """A FinEdge request failed."""


class FinEdgePlanError(FinEdgeError):
    """The FinEdge plan does not include this symbol or endpoint (HTTP 401)."""


def resolve_finedge_token(config=None) -> str | None:
    """Return the FinEdge token from config ([API_KEYS] finedge_api_token) or env."""
    if config is not None:
        try:
            value = config.get("API_KEYS", "finedge_api_token", fallback=None)
            if value and not value.strip().upper().startswith("YOUR_"):
                return value.strip()
        except Exception:
            pass
    return os.getenv(TOKEN_ENV_VAR) or None


def normalize_symbol(symbol: str) -> str:
    """Strip exchange suffixes such as .NS / .BO used by other data sources."""
    s = symbol.strip().upper()
    for suffix in (".NS", ".NSE", ".BO", ".BSE"):
        if s.endswith(suffix):
            return s[: -len(suffix)]
    return s


class FinEdgeClient:
    """Small FinEdge REST client with in-memory caching.

    The token goes in the ``token`` query parameter (FinEdge's only auth
    method). It is never included in error messages.
    """

    def __init__(self, token: str | None = None, base_url: str = FINEDGE_BASE_URL,
                 timeout: float = 30, pause: float = 0.2, session: requests.Session | None = None):
        self.token = token or os.getenv(TOKEN_ENV_VAR)
        if not self.token:
            raise FinEdgeError(
                f"No FinEdge API token. Set {TOKEN_ENV_VAR} or add finedge_api_token to config.ini."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.pause = pause
        self.session = session or requests.Session()
        self._cache: dict[tuple, Any] = {}

    def _mask(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text

    def get(self, path: str, **params) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        key = (path, tuple(sorted(clean.items())))
        if key in self._cache:
            return self._cache[key]
        query = dict(clean, token=self.token)
        last_error = None
        for attempt in range(3):
            try:
                resp = self.session.get(self.base_url + path, params=query, timeout=self.timeout)
            except requests.RequestException as e:
                last_error = FinEdgeError(self._mask(f"FinEdge request to {path} failed: {e}"))
                time.sleep(1 + attempt)
                continue
            if resp.status_code == 401:
                raise FinEdgePlanError(
                    f"FinEdge plan does not allow {path} ({self._mask(resp.text.strip())[:120]})"
                )
            if resp.status_code >= 500 or resp.status_code == 429:
                last_error = FinEdgeError(f"FinEdge {path} returned HTTP {resp.status_code}")
                time.sleep(1 + attempt)
                continue
            if resp.status_code != 200:
                raise FinEdgeError(
                    f"FinEdge {path} returned HTTP {resp.status_code}: {self._mask(resp.text.strip())[:200]}"
                )
            try:
                data = resp.json()
            except ValueError as e:
                raise FinEdgeError(f"FinEdge {path} returned non-JSON content") from e
            self._cache[key] = data
            if self.pause:
                time.sleep(self.pause)
            return data
        raise last_error or FinEdgeError(f"FinEdge request to {path} failed")

    # --- Endpoint helpers -------------------------------------------------
    def financials(self, symbol, statement_code, period="annual", statement_type="c"):
        data = self.get(f"/api/v1/financials/{symbol}", statement_type=statement_type,
                        statement_code=statement_code, period=period)
        return (data or {}).get("financials") or []

    def ratios(self, symbol, ratio_type, statement_type="c"):
        data = self.get(f"/api/v1/ratios/{symbol}", statement_type=statement_type, ratio_type=ratio_type)
        return (data or {}).get("ratios") or []

    def annual_price_ratios(self, symbol, statement_type="c"):
        data = self.get(f"/api/v1/annual-price-ratios/{symbol}", statement_type=statement_type)
        return (data or {}).get("price_ratios") or []

    def daily_price_ratios(self, symbol, statement_type="c", from_year=None):
        data = self.get(f"/api/v1/daily-price-ratios/{symbol}", statement_type=statement_type,
                        **({"from": from_year} if from_year else {}))
        return (data or {}).get("price_ratios") or []

    def quote(self, symbol):
        data = self.get("/api/v1/quote", symbol=symbol) or {}
        return data.get(symbol) or (next(iter(data.values())) if data else {})

    def profile(self, symbol):
        return self.get(f"/api/v1/company-profile/{symbol}") or {}

    def peers(self, symbol, group="sector"):
        return (self.get(f"/api/v1/peers/{symbol}", group=group) or {}).get("peers") or []

    def daily_quotes(self, symbol, from_year=None, to_year=None):
        params = {}
        if from_year:
            params["from"] = from_year
        if to_year:
            params["to"] = to_year
        return (self.get(f"/api/v1/daily-quotes/{symbol}", **params) or {}).get("price") or []

    def shareholding_pattern(self, symbol, period="quarterly"):
        return self.get(f"/api/v1/shareholdings/pattern/{symbol}", period=period) or {}

    def dividends(self, symbol):
        return (self.get(f"/api/v1/dividend/{symbol}") or {}).get("dividend") or []

    def announcements(self, symbol, from_date, to_date):
        return self.get("/api/v1/corp-announcements", symbol=symbol,
                        from_date=from_date, to_date=to_date) or []

    def index_history(self, index_symbol, from_date, to_date):
        data = self.get("/api/v1/index/market-price/historical", index_symbol=index_symbol,
                        from_date=from_date, to_date=to_date) or {}
        return data.get("rows") or []


# =============================================================================
# Helpers
# =============================================================================
def _num(value) -> float:
    """Return a float, or NaN for None / non-numeric values."""
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _nonzero(value) -> float:
    """Return a float, treating 0 (FinEdge's 'missing' marker) as NaN."""
    v = _num(value)
    return np.nan if v == 0 else v


def _first(row: dict, *keys) -> float:
    """First non-zero numeric value among keys."""
    for k in keys:
        v = _nonzero(row.get(k))
        if pd.notna(v):
            return v
    return np.nan


def _sum(row: dict, *keys) -> float:
    vals = [_num(row.get(k)) for k in keys]
    vals = [v for v in vals if pd.notna(v)]
    return float(sum(vals)) if vals else np.nan


def _div(a, b) -> float:
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return float(a) / float(b)


def _period_date(value) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(str(int(value)), format="%Y%m%d")
    except (TypeError, ValueError):
        return None


def _base_frame(rows: list[dict]) -> list[dict]:
    """Sort newest first and drop rows without a period end."""
    rows = [r for r in rows if r.get("period_end")]
    return sorted(rows, key=lambda r: r["period_end"], reverse=True)


def _meta(r: dict, period: str) -> dict:
    date = _period_date(r.get("period_end"))
    year = int(r.get("year") or (date.year if date is not None else 0))
    return {
        "date": date,
        "year": year,
        "calendarYear": year,
        "period": "FY" if period == "annual" else period,
        "reportedCurrency": "INR",
        "fillingDate": _period_date(r.get("result_date")),
    }


def _is_bank_bs_row(r: dict) -> bool:
    """Bank balance sheets have non-zero deposits and advances (other schemas carry them as 0)."""
    return pd.notna(_nonzero(r.get("deposits"))) and pd.notna(_nonzero(r.get("advances")))


def is_bank_statement(pl_rows: list[dict], bs_rows: list[dict] | None = None) -> bool:
    """True when FinEdge returned the bank layout (interest earned/expended, deposits)."""
    sample = (pl_rows or [{}])[0]
    if "interestExpended" in sample and "revenueFromOperations" not in sample:
        return True
    if bs_rows and _is_bank_bs_row(bs_rows[0]):
        return True
    return False


# =============================================================================
# Statement normalisation (FinEdge rows -> FMP-like DataFrames)
# =============================================================================
def normalize_income_statement(rows: list[dict], period: str = "annual") -> pd.DataFrame:
    """Map FinEdge P&L rows to FMP income-statement columns (absolute rupees)."""
    out = []
    bank = is_bank_statement(rows)
    for r in _base_frame(rows):
        rec = _meta(r, period)
        if bank:
            interest_earned = _num(r.get("interestEarned"))
            interest_expended = _num(r.get("interestExpended"))
            other_income = _num(r.get("otherIncome"))
            opex = _sum(r, "employeesCost", "otherOperatingExpenses")
            nii = interest_earned - interest_expended
            total_income = _num(r.get("income"))
            ppop = total_income - _num(r.get("expenditureExcludingProvisions"))
            pbt = _num(r.get("profitLossBeforeTax"))
            pat_total = _num(r.get("profitLossForThePeriod"))
            rec.update({
                "revenue": total_income,                      # FMP convention for banks
                "interestIncome": interest_earned,
                "interestExpense": interest_expended,
                "netInterestIncome": nii,
                "otherIncome": other_income,
                "operatingExpenses": opex,
                "costOfRevenue": interest_expended,
                "grossProfit": nii + other_income,           # net total income
                "sellingGeneralAndAdministrativeExpenses": opex,
                "preProvisionOperatingProfit": ppop,
                "provisions": _num(r.get("provisionsForLoanLoss")),
                "ebitda": np.nan,                             # not meaningful for banks
                "operatingIncome": ppop,
                "incomeBeforeTax": pbt,
                "incomeTaxExpense": _num(r.get("taxExpense")),
                "netIncome": _first(r, "profitOrLossAttributableToOwners") if pd.notna(
                    _first(r, "profitOrLossAttributableToOwners")) else pat_total,
                "minorityInterest": _num(r.get("profitLossOfMinorityInterest")),
                "eps": _nonzero(r.get("eps")),
                "epsdiluted": _nonzero(r.get("eps")),
                "weightedAverageShsOutDil": _nonzero(r.get("dilutedOutstandingShares")),
                "grossNPA": _nonzero(r.get("grossNonPerformingAssets")),
                "netNPA": _nonzero(r.get("nonPerformingAssets")),
                "grossNPARatio": _nonzero(r.get("percentageOfGrossNpa")),
                "netNPARatio": _nonzero(r.get("percentageOfNpa")),
                "cet1Ratio": _nonzero(r.get("cET1Ratio")),
            })
        else:
            revenue = _num(r.get("revenueFromOperations"))
            cogs = _first(r, "costofGoodsSold")
            if pd.isna(cogs):
                cogs = _sum(r, "costOfMaterialsConsumed", "purchasesOfStockInTrade", "changesInInventories")
            finance = _num(r.get("financeCosts"))
            dna = _num(r.get("depreciationAndAmortisation"))
            expenses = _num(r.get("expenses"))
            ebitda = revenue - (expenses - finance - dna)
            sga = _sum(r, "employeeBenefitExpense", "otherExpenses")
            owners = _first(r, "profitOrLossAttributableToOwners")
            rec.update({
                "revenue": revenue,
                "costOfRevenue": cogs,
                "grossProfit": revenue - cogs,
                "employeeBenefitExpense": _num(r.get("employeeBenefitExpense")),
                "otherExpenses": _num(r.get("otherExpenses")),
                "sellingGeneralAndAdministrativeExpenses": sga,
                "operatingExpenses": sga,
                "costAndExpenses": expenses,
                "ebitda": ebitda,
                "depreciationAndAmortization": dna,
                "operatingIncome": ebitda - dna,
                "interestExpense": finance,
                "otherIncome": _num(r.get("otherIncome")),
                "exceptionalItems": _num(r.get("exceptionalItemsBeforeTax")),
                "incomeBeforeTax": _num(r.get("profitBeforeTax")),
                "incomeTaxExpense": _num(r.get("taxExpense")),
                "shareOfAssociates": _num(r.get("profitOrLossOfAssociates")),
                "netIncome": owners if pd.notna(owners) else _num(r.get("profitLossForPeriod")),
                "netIncomeIncludingMinority": _num(r.get("profitLossForPeriod")),
                "discontinuedOperations": _num(r.get("profitLossFromDiscontinuedOperationsAfterTax")),
                "eps": _nonzero(r.get("eps")),
                "epsdiluted": _nonzero(r.get("eps")),
                "weightedAverageShsOutDil": _nonzero(r.get("dilutedOutstandingShares")),
            })
        rec["grossProfitRatio"] = _div(rec.get("grossProfit"), rec.get("revenue"))
        rec["ebitdaratio"] = _div(rec.get("ebitda"), rec.get("revenue"))
        rec["netIncomeRatio"] = _div(rec.get("netIncome"), rec.get("revenue"))
        out.append(rec)
    return pd.DataFrame(out)


def normalize_balance_sheet(rows: list[dict], period: str = "annual") -> pd.DataFrame:
    """Map FinEdge balance-sheet rows to FMP balance-sheet columns."""
    out = []
    for r in _base_frame(rows):
        rec = _meta(r, period)
        if _is_bank_bs_row(r):  # bank layout
            equity = _sum(r, "capital", "reserves")
            debt = _num(r.get("borrowings"))
            cash = _sum(r, "cashAndBalancesWithRBI", "bankBalancesAndMoneyAtCallAndShortNotice")
            rec.update({
                "totalAssets": _num(r.get("assets")),
                "totalStockholdersEquity": equity,
                "totalEquity": equity,
                "totalLiabilities": _num(r.get("assets")) - equity,
                "totalDebt": debt,
                "cashAndCashEquivalents": cash,
                "cashAndShortTermInvestments": cash,
                "netDebt": np.nan,
                "deposits": _num(r.get("deposits")),
                "advances": _num(r.get("advances")),
                "investments": _num(r.get("investments")),
                "goodwill": _num(r.get("goodwillOnConsolidation")),
                "totalCurrentAssets": np.nan,
                "totalCurrentLiabilities": np.nan,
            })
        else:
            debt = _sum(r, "borrowingsCurrent", "borrowingsNoncurrent")
            cash = _num(r.get("cashAndCashEquivalents"))
            cash_st = _sum(r, "cashAndCashEquivalents", "otherBankBalances", "currentInvestments")
            owners = _first(r, "equityAttributableToOwnersOfParent")
            rec.update({
                "totalAssets": _num(r.get("assets")),
                "totalLiabilities": _num(r.get("liabilities")),
                "totalStockholdersEquity": owners if pd.notna(owners) else _num(r.get("totalEquity")),
                "totalEquity": _num(r.get("totalEquity")),
                "minorityInterest": _num(r.get("nonControllingInterest")),
                "totalDebt": debt,
                "shortTermDebt": _num(r.get("borrowingsCurrent")),
                "longTermDebt": _num(r.get("borrowingsNoncurrent")),
                "cashAndCashEquivalents": cash,
                "shortTermInvestments": _num(r.get("currentInvestments")),
                "cashAndShortTermInvestments": cash_st,
                "netDebt": debt - cash_st,
                "totalCurrentAssets": _num(r.get("currentAssets")),
                "totalCurrentLiabilities": _num(r.get("currentLiabilities")),
                "inventory": _num(r.get("inventories")),
                "netReceivables": _num(r.get("tradeReceivablesCurrent")),
                "accountPayables": _num(r.get("tradePayablesCurrent")),
                "propertyPlantEquipmentNet": _num(r.get("propertyPlantAndEquipment")),
                "goodwill": _num(r.get("goodwill")),
            })
        out.append(rec)
    return pd.DataFrame(out)


def normalize_cash_flow(rows: list[dict], period: str = "annual") -> pd.DataFrame:
    """Map FinEdge cash-flow rows to FMP cash-flow columns."""
    out = []
    for r in _base_frame(rows):
        rec = _meta(r, period)
        ocf = _num(r.get("cashFlowsFromOperatingActivities"))
        capex = _first(r, "purchaseOfFixed&IntangibleAssets", "purchaseOfPPEClassifiedAsInvesting",
                       "purchaseOfTangibleAssetsClassifiedAsInvesting")
        capex = -abs(capex) if pd.notna(capex) else 0.0
        rec.update({
            "operatingCashFlow": ocf,
            "netCashProvidedByOperatingActivities": ocf,
            "capitalExpenditure": capex,
            "freeCashFlow": ocf + capex if pd.notna(ocf) else np.nan,
            "netCashUsedForInvestingActivites": _num(r.get("cashFlowsFromInvestingActivities")),
            "netCashUsedProvidedByFinancingActivities": _num(r.get("cashFlowsFromFinancingActivities")),
            "dividendsPaid": _num(r.get("dividendsPaidClassifiedAsFinancing")),
            "netChangeInCash": _num(r.get("netCashFlow")),
        })
        out.append(rec)
    return pd.DataFrame(out)


def _annual_rows(rows: list[dict]) -> dict[int, dict]:
    """Ratio rows keyed by fiscal year, ignoring TTM and interim (Sep/Dec) rows."""
    result = {}
    for r in rows:
        header = str(r.get("header", ""))
        if header == "TTM" or not r.get("year"):
            continue
        if header and not header.startswith(("Mar", "Jun", "Sep", "Dec")):
            continue
        year = int(r["year"])
        # Prefer the fiscal-year-end row (FinEdge also returns interim rows for some symbols)
        if year in result and not header.startswith("Mar"):
            continue
        result[year] = r
    return result


def _ttm_row(rows: list[dict]) -> dict:
    return next((r for r in rows if r.get("header") == "TTM"), {})


def build_ratios_frames(ratio_rows: dict[str, list[dict]], price_rows: list[dict],
                        income_df: pd.DataFrame, balance_df: pd.DataFrame,
                        cash_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build FMP-like ``ratios`` and ``key_metrics`` DataFrames (one row per fiscal year)."""
    pr = _annual_rows(ratio_rows.get("pr", []))
    le = _annual_rows(ratio_rows.get("le", []))
    li = _annual_rows(ratio_rows.get("li", []))
    ef = _annual_rows(ratio_rows.get("ef", []))
    px = _annual_rows(price_rows)

    inc = income_df.set_index("year") if income_df is not None and not income_df.empty else pd.DataFrame()
    bal = balance_df.set_index("year") if balance_df is not None and not balance_df.empty else pd.DataFrame()
    cfs = cash_df.set_index("year") if cash_df is not None and not cash_df.empty else pd.DataFrame()

    is_bank = bool(not bal.empty and "deposits" in bal.columns and bal["deposits"].notna().any())
    years = sorted(set(inc.index) if not inc.empty else set(pr), reverse=True)
    ratios, key_metrics = [], []
    for year in years:
        p, l, q, e, x = pr.get(year, {}), le.get(year, {}), li.get(year, {}), ef.get(year, {}), px.get(year, {})
        date = inc.loc[year, "date"] if year in inc.index else pd.Timestamp(year=year, month=3, day=31)

        def _get(df, col):
            return _num(df.loc[year, col]) if (not df.empty and year in df.index and col in df.columns) else np.nan

        debt = _get(bal, "totalDebt")
        ocf = _get(cfs, "operatingCashFlow")
        ratios.append({
            "date": date, "year": year, "calendarYear": year,
            "grossProfitMargin": _nonzero(p.get("grossMargin")),
            "ebitdaMargin": _get(inc, "ebitdaratio"),
            "operatingProfitMargin": _nonzero(p.get("operatingMargin")),
            "pretaxProfitMargin": _nonzero(p.get("preTaxMargin")),
            "netProfitMargin": _div(_get(inc, "netIncome"), _get(inc, "revenue")),
            "effectiveTaxRate": _nonzero(p.get("effectiveTaxRate")),
            "returnOnEquity": _num(p.get("returnOnEquity")),
            "returnOnAssets": _num(p.get("returnOnAsset")),
            "returnOnCapitalEmployed": _num(p.get("returnOnCapital")),
            "netInterestMargin": _nonzero(p.get("netInterestMargin")),
            "debtEquityRatio": _num(l.get("totalDebtToEquity")),
            "debtRatio": _num(l.get("totalDebttoAssets")),
            "longTermDebtToCapitalization": _num(l.get("longTermDebtToTotalAssets")),
            "financialLeverage": _num(l.get("financialLeverage")),
            "interestCoverage": _nonzero(q.get("interestCoverage")),
            "currentRatio": _nonzero(q.get("currentRatio")),
            "quickRatio": _nonzero(q.get("quickRatio")),
            "cashFlowToDebtRatio": _div(ocf, debt) if pd.notna(debt) and debt > 0 and not is_bank else np.nan,
            "assetTurnover": _num(e.get("assetTurnover")),
            "inventoryTurnover": _nonzero(e.get("inventoryTurnover")),
            "receivablesTurnover": _nonzero(e.get("receivableTurnover")),
            "cashConversionCycle": _num(e.get("cashConversionCycle")),
            "priceEarningsRatio": _nonzero(x.get("pe")),
            "priceToBookRatio": _nonzero(x.get("pb")),
            "priceToSalesRatio": _nonzero(x.get("ps")),
            "priceToFreeCashFlowsRatio": _nonzero(x.get("pfcf")),
        })

        # Key metrics: EV from the year's average price x diluted shares
        avg_price = _nonzero(x.get("average_price"))
        shares = _get(inc, "weightedAverageShsOutDil")
        market_cap = avg_price * shares if pd.notna(avg_price) and pd.notna(shares) else np.nan
        cash_st = _get(bal, "cashAndShortTermInvestments")
        ebitda = _get(inc, "ebitda")
        ev = np.nan
        if pd.notna(market_cap) and not is_bank:
            ev = market_cap + (debt if pd.notna(debt) else 0) - (cash_st if pd.notna(cash_st) else 0)
        key_metrics.append({
            "date": date, "year": year, "calendarYear": year,
            "averagePrice": avg_price,
            "marketCap": market_cap,
            "enterpriseValue": ev,
            "enterpriseValueOverEBITDA": _div(ev, ebitda) if pd.notna(ebitda) and ebitda > 0 else np.nan,
            "peRatio": _nonzero(x.get("pe")),
            "pbRatio": _nonzero(x.get("pb")),
            "priceToSalesRatio": _nonzero(x.get("ps")),
            "pfcfRatio": _nonzero(x.get("pfcf")),
            "netIncomePerShare": _get(inc, "eps"),
            "roe": _num(p.get("returnOnEquity")),
        })
    return pd.DataFrame(ratios), pd.DataFrame(key_metrics)


# =============================================================================
# Public API used by the pipeline
# =============================================================================
def detect_statement_type(client: FinEdgeClient, symbol: str, requested: str = "auto") -> tuple[str, bool]:
    """Pick standalone ('s') or consolidated ('c') and report whether it is a bank.

    Companies default to consolidated. Banks default to standalone, because
    FinEdge only fills asset-quality and capital fields (GNPA, NNPA, CET1) in
    standalone, and consolidated bank figures mix in insurance subsidiaries.
    """
    pl_c = client.financials(symbol, "pl", "annual", "c") if requested in ("auto", "c") else []
    bank = is_bank_statement(pl_c) if pl_c else False
    if requested in ("s", "c"):
        if requested == "s":
            bank = is_bank_statement(client.financials(symbol, "pl", "annual", "s"))
        return requested, bank
    if not pl_c:
        return "s", is_bank_statement(client.financials(symbol, "pl", "annual", "s"))
    return ("s" if bank else "c"), bank


def get_finedge_comprehensive_financial_data(symbol: str, client: FinEdgeClient, period: str = "annual",
                                             limit: int = 5, statement_type: str = "auto") -> dict:
    """FinEdge equivalent of ``get_comprehensive_financial_data``.

    Returns the same keys (income_statement, balance_sheet, cash_flow, ratios,
    key_metrics) plus ``meta`` with statement type, bank flag and data warnings.
    """
    symbol = normalize_symbol(symbol)
    st, bank = detect_statement_type(client, symbol, statement_type)
    print(f"Fetching FinEdge data for {symbol} ({'standalone' if st == 's' else 'consolidated'}"
          f"{', bank layout' if bank else ''})...")

    pl = client.financials(symbol, "pl", period, st)
    bs = client.financials(symbol, "bs", period, st)
    cf = client.financials(symbol, "cf", period, st)
    income_df = normalize_income_statement(pl, period)
    balance_df = normalize_balance_sheet(bs, period)
    cash_df = normalize_cash_flow(cf, period)

    ratio_rows = {}
    for rt in ("pr", "le", "li", "ef"):
        try:
            ratio_rows[rt] = client.ratios(symbol, rt, st)
        except FinEdgeError as e:
            print(f"Warning: FinEdge ratios ({rt}) unavailable: {e}")
            ratio_rows[rt] = []
    try:
        price_rows = client.annual_price_ratios(symbol, st)
    except FinEdgeError as e:
        print(f"Warning: FinEdge annual price ratios unavailable: {e}")
        price_rows = []

    ratios_df, key_metrics_df = build_ratios_frames(ratio_rows, price_rows, income_df, balance_df, cash_df)

    def _limit(df):
        if df is None or df.empty:
            return df
        return df.sort_values("date", ascending=False).head(limit).reset_index(drop=True)

    warnings = collect_data_warnings(symbol, _limit(income_df), cash_df, bank, limit, _limit(balance_df))
    return {
        "income_statement": _limit(income_df),
        "balance_sheet": _limit(balance_df),
        "cash_flow": _limit(cash_df),
        "ratios": _limit(ratios_df),
        "key_metrics": _limit(key_metrics_df),
        "meta": {
            "data_source": "FinEdge API",
            "symbol": symbol,
            "statement_type": "standalone" if st == "s" else "consolidated",
            "is_bank": bank,
            "currency": "INR",
            "units": "absolute rupees",
            "ttm_ratios": {rt: _ttm_row(rows) for rt, rows in ratio_rows.items()},
            "warnings": warnings,
        },
    }


def collect_data_warnings(symbol: str, income_df: pd.DataFrame, cash_df: pd.DataFrame,
                          bank: bool, limit: int, balance_df: pd.DataFrame | None = None) -> list[str]:
    """Known FinEdge caveats worth showing to the analyst."""
    w = []
    # One-off items that distort EPS and profit growth
    if income_df is not None and "discontinuedOperations" in income_df.columns:
        for _, r in income_df.iterrows():
            d, ni = r.get("discontinuedOperations"), r.get("netIncome")
            if pd.notna(d) and pd.notna(ni) and ni and abs(d) > 0.05 * abs(ni):
                w.append(f"FY{int(r['year'])} net profit and EPS include {d / CRORE:,.0f} crore rupees from "
                         f"discontinued operations; that year's profit growth is not like-for-like.")
    # Balance-sheet jumps (mergers, acquisitions)
    if bank and balance_df is not None and "advances" in balance_df.columns:
        b = balance_df.sort_values("year")
        growth = b["advances"].pct_change()
        for year, g in zip(b["year"], growth):
            if pd.notna(g) and g > 0.40:
                w.append(f"Advances grew {g:.0%} in FY{int(year)}, which points to a merger or acquisition; "
                         f"growth rates for that year are not like-for-like.")
    if not bank:
        w.append("EBITDA, gross profit and SG&A are derived: FinEdge has no such fields "
                 "(EBITDA excludes other income and exceptional items; SG&A = employee + other expenses).")
    if cash_df is not None and len(cash_df) < limit:
        w.append(f"Only {len(cash_df)} years of cash-flow data are available from FinEdge.")
    if symbol == "ITC":
        w.append("ITC revenue includes excise duty and GST on cigarettes from 1 Feb 2026 (per company notes); "
                 "FY26 onward revenue growth and margins are not comparable with earlier years.")
    if bank:
        w.append("Bank figures use standalone statements: FinEdge reports GNPA, NNPA and CET1 only in standalone.")
    return w


def get_finedge_price_history(symbol: str, client: FinEdgeClient, years: int = 2) -> pd.DataFrame:
    """Adjusted daily OHLCV, oldest first, with zero-price rows (holidays) dropped."""
    symbol = normalize_symbol(symbol)
    this_year = dt.date.today().year
    rows = client.daily_quotes(symbol, from_year=this_year - years + 1, to_year=this_year)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"quote_date": "date", "open_price": "open", "high_price": "high",
                            "low_price": "low", "close_price": "close"})
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def get_finedge_technical_indicators(symbol: str, client: FinEdgeClient) -> dict:
    """SMA/RSI/MACD/volume signals from FinEdge prices (same output as the FMP version)."""
    from .market_data_api import compute_technical_indicators
    try:
        df = get_finedge_price_history(symbol, client, years=2).tail(250)
    except FinEdgeError as e:
        print(f"⚠️ Could not fetch FinEdge prices for technicals: {e}")
        df = pd.DataFrame()
    return compute_technical_indicators(df, symbol)


def _beta_vs_nifty(symbol: str, client: FinEdgeClient, prices: pd.DataFrame) -> float | None:
    """1-year daily beta against the NIFTY 50."""
    if prices is None or len(prices) < 120:
        return None
    end = prices["date"].max()
    start = end - pd.Timedelta(days=365)
    try:
        idx = pd.DataFrame(client.index_history(NIFTY50_SYMBOL, start.strftime("%Y-%m-%d"),
                                                end.strftime("%Y-%m-%d")))
    except FinEdgeError:
        return None
    if idx.empty:
        return None
    idx["date"] = pd.to_datetime(idx["quote_date"])
    merged = prices[prices["date"] >= start][["date", "close"]].merge(
        idx[["date", "close_price"]], on="date", how="inner").sort_values("date")
    rets = merged[["close", "close_price"]].pct_change().dropna()
    if len(rets) < 100 or rets["close_price"].var() == 0:
        return None
    return float(rets["close"].cov(rets["close_price"]) / rets["close_price"].var())


def _free_float(pattern: dict) -> float | None:
    """100 - promoter holding (%) from the latest shareholding pattern."""
    rows = pattern.get("rows") or []
    cols = pattern.get("columns") or []
    if not rows or not cols:
        return None
    latest = cols[-1]
    promoter = sum(_num(r.get("data", {}).get(latest)) or 0 for r in rows
                   if "romoter" in str(r.get("name", "")) and "non-promoter" not in str(r.get("name", "")).lower())
    return round(100.0 - promoter, 2)


def get_finedge_company_metrics(symbol: str, client: FinEdgeClient, financial_data: dict | None = None) -> dict:
    """FinEdge equivalent of ``get_comprehensive_company_metrics``.

    ``market_cap`` is in rupee crore. Analyst target price and rating are not
    available from FinEdge and stay None / 'N/A'.
    """
    symbol = normalize_symbol(symbol)
    metrics = {k: None for k in (
        "share_price", "target_price", "market_cap", "volume", "fwd_pe", "pb_ratio", "dividend_yield",
        "free_float", "roe", "net_debt_to_equity", "rating", "beta", "sector", "industry", "exchange",
        "52w_range", "shares_outstanding")}
    metrics["rating"] = "N/A"
    metrics["exchange"] = "NSE"

    try:
        q = client.quote(symbol)
        metrics["share_price"] = _nonzero(q.get("current_price")) if q else None
        metrics["market_cap"] = _nonzero(q.get("market_cap"))  # ₹ crore
        # /quote 'shares' is unreliable (pre-bonus counts for RELIANCE/HDFCBANK); derive from market cap
        price, mcap = metrics["share_price"], metrics["market_cap"]
        if pd.notna(price) and pd.notna(mcap) and price:
            metrics["shares_outstanding"] = mcap * CRORE / price
        lo, hi = _nonzero(q.get("low52")), _nonzero(q.get("high52"))
        if pd.notna(lo) and pd.notna(hi):
            metrics["52w_range"] = (lo, hi)
    except FinEdgeError as e:
        print(f"Warning: FinEdge quote unavailable: {e}")

    try:
        prof = client.profile(symbol)
        metrics["sector"] = prof.get("sector") or prof.get("macro_sector")
        metrics["industry"] = prof.get("industry")
        metrics["description"] = prof.get("description")
        metrics["website"] = prof.get("website")
        if metrics["market_cap"] is None or pd.isna(metrics["market_cap"]):
            metrics["market_cap"] = _nonzero(prof.get("market_cap"))
    except FinEdgeError as e:
        print(f"Warning: FinEdge profile unavailable: {e}")

    st = "c"
    if financial_data and financial_data.get("meta", {}).get("statement_type") == "standalone":
        st = "s"
    try:
        daily = client.daily_price_ratios(symbol, st, from_year=dt.date.today().year - 1)
        daily = sorted([d for d in daily if pd.notna(_nonzero(d.get("pe")))],
                       key=lambda d: d["quote_date"])
        if daily:
            metrics["fwd_pe"] = _nonzero(daily[-1].get("pe"))   # trailing P/E (FinEdge has no forward P/E)
            metrics["pb_ratio"] = _nonzero(daily[-1].get("pb"))
    except FinEdgeError as e:
        print(f"Warning: FinEdge daily price ratios unavailable: {e}")

    prices = pd.DataFrame()
    try:
        prices = get_finedge_price_history(symbol, client, years=2)
        if not prices.empty:
            metrics["volume"] = float(prices["volume"].tail(60).mean()) / 1e6
            if metrics["share_price"] is None or pd.isna(metrics["share_price"]):
                metrics["share_price"] = float(prices["close"].iloc[-1])
    except FinEdgeError as e:
        print(f"Warning: FinEdge price history unavailable: {e}")
    metrics["beta"] = _beta_vs_nifty(symbol, client, prices)

    try:
        price = metrics["share_price"]
        cutoff = dt.date.today() - dt.timedelta(days=365)
        total = 0.0
        for d in client.dividends(symbol):
            try:
                when = dt.datetime.strptime(d["date"], "%d-%b-%Y").date()
            except (KeyError, ValueError):
                continue
            if when >= cutoff:
                total += _num(d.get("adj_amount") or d.get("amount")) or 0
        if price and total:
            metrics["dividend_yield"] = total / price * 100
    except FinEdgeError as e:
        print(f"Warning: FinEdge dividends unavailable: {e}")

    try:
        metrics["free_float"] = _free_float(client.shareholding_pattern(symbol))
    except FinEdgeError as e:
        print(f"Warning: FinEdge shareholding unavailable: {e}")

    if financial_data:
        meta = financial_data.get("meta", {})
        ttm = meta.get("ttm_ratios", {}).get("pr", {})
        if ttm.get("returnOnEquity") is not None:
            metrics["roe"] = float(ttm["returnOnEquity"]) * 100
        bal = financial_data.get("balance_sheet")
        if bal is not None and not bal.empty and not meta.get("is_bank"):
            latest = bal.iloc[0]
            if pd.notna(latest.get("netDebt")) and latest.get("totalStockholdersEquity"):
                metrics["net_debt_to_equity"] = latest["netDebt"] / latest["totalStockholdersEquity"] * 100

    metrics["data_source"] = "FinEdge API"
    print(f"Successfully fetched FinEdge metrics for {symbol}")
    return metrics


def get_finedge_company_news(symbol: str, client: FinEdgeClient, days_back: int = 30,
                             limit: int = 50) -> list[dict] | None:
    """Exchange announcements in the FMP news format (FinEdge has no news feed)."""
    symbol = normalize_symbol(symbol)
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    try:
        rows = client.announcements(symbol, start.isoformat(), end.isoformat())
    except FinEdgeError as e:
        print(f"Warning: FinEdge announcements unavailable: {e}")
        return None
    news = []
    for r in sorted(rows, key=lambda x: x.get("announcement_date", ""), reverse=True)[:limit]:
        category = r.get("category") or "Announcement"
        description = r.get("description") or ""
        news.append({
            "symbol": symbol,
            "title": f"{category}: {description}"[:300],
            "publishedDate": r.get("announcement_date"),
            "text": description,
            "site": "NSE/BSE filing",
            "url": r.get("pdf_file_link"),
            "category": category,
        })
    return news or None


def get_finedge_peers(symbol: str, client: FinEdgeClient, group: str = "sector", max_peers: int = 5) -> list[str]:
    """Peer symbols from FinEdge (excluding the company itself)."""
    symbol = normalize_symbol(symbol)
    try:
        peers = client.peers(symbol, group)
    except FinEdgeError as e:
        print(f"Warning: FinEdge peers unavailable: {e}")
        return []
    return [p for p in peers if p != symbol][:max_peers]


def filter_valid_peers(symbols: list[str], client: FinEdgeClient) -> tuple[list[str], list[str]]:
    """Split typed peer symbols into (recognised by FinEdge, ignored).

    Guards against free text such as "sector peers" being treated as tickers.
    """
    valid, ignored = [], []
    for raw in symbols:
        sym = normalize_symbol(raw)
        try:
            known = bool(client.profile(sym))
        except FinEdgePlanError:
            known = True  # a real symbol that the plan doesn't cover; reported later as skipped
        except FinEdgeError:
            known = False
        (valid if known else ignored).append(sym)
    return valid, ignored


def combine_peer_financial_data_finedge(tickers: list[str], client: FinEdgeClient,
                                        years_limit: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Peer EBITDA and EV/EBITDA pivots (index = fiscal year, columns = ticker).

    Symbols outside the FinEdge plan are skipped and returned in the third value.
    """
    ebitda, ev_ebitda, skipped = {}, {}, []
    for t in tickers:
        sym = normalize_symbol(t)
        try:
            data = get_finedge_comprehensive_financial_data(sym, client, limit=years_limit)
        except FinEdgePlanError:
            print(f"Skipping peer {sym}: not included in the FinEdge plan (HTTP 401).")
            skipped.append(sym)
            continue
        except FinEdgeError as e:
            print(f"Skipping peer {sym}: {e}")
            skipped.append(sym)
            continue
        inc, km = data["income_statement"], data["key_metrics"]
        if inc is not None and not inc.empty:
            ebitda[sym] = inc.set_index("year")["ebitda"]
        if km is not None and not km.empty:
            ev_ebitda[sym] = km.set_index("year")["enterpriseValueOverEBITDA"]
    df_e = pd.DataFrame(ebitda).sort_index().dropna(how="all") if ebitda else pd.DataFrame()
    df_ev = pd.DataFrame(ev_ebitda).sort_index().dropna(how="all") if ev_ebitda else pd.DataFrame()
    for df in (df_e, df_ev):
        if not df.empty:
            df.index.name = "year"
    return df_e, df_ev, skipped


def combine_peer_bank_data_finedge(tickers: list[str], client: FinEdgeClient) -> tuple[pd.DataFrame, list[str]]:
    """Latest-year bank comparison (P/B, ROE, ROA, NIM, GNPA, NNPA, CET1) for bank peers."""
    rows, skipped = [], []
    for t in tickers:
        sym = normalize_symbol(t)
        try:
            data = get_finedge_comprehensive_financial_data(sym, client, limit=1, statement_type="s")
        except FinEdgeError as e:
            print(f"Skipping bank peer {sym}: {e}")
            skipped.append(sym)
            continue
        inc, rat = data["income_statement"], data["ratios"]
        if inc is None or inc.empty:
            continue
        i, r = inc.iloc[0], (rat.iloc[0] if rat is not None and not rat.empty else {})
        rows.append({
            "ticker": sym, "year": int(i["year"]),
            "P/B": r.get("priceToBookRatio"), "ROE": r.get("returnOnEquity"),
            "ROA": r.get("returnOnAssets"), "NIM": r.get("netInterestMargin"),
            "GNPA": i.get("grossNPARatio"), "NNPA": i.get("netNPARatio"), "CET1": i.get("cet1Ratio"),
        })
    return pd.DataFrame(rows), skipped


# =============================================================================
# Bank analysis table
# =============================================================================
BANK_METRICS_ORDER = [
    "Net Interest Income", "NII Growth", "Other Income", "Total Income", "Operating Expenses",
    "Cost-to-Income", "Pre-Provision Profit", "Provisions", "Credit Cost", "Net Profit",
    "Net Profit Growth", "Advances", "Advances Growth", "Deposits", "Deposits Growth",
    "Credit-Deposit Ratio", "NIM", "ROA", "ROE", "Gross NPA", "Net NPA", "CET1 Ratio",
    "EPS", "Book Value per Share", "PE Ratio", "PB Ratio", "Revenue", "EBITDA",
]


def _pct(v, digits=1):
    return f"{v * 100:.{digits}f}%" if v is not None and pd.notna(v) else "N/A"


def build_bank_metrics_table(financial_data: dict) -> pd.DataFrame:
    """Bank analysis table in the same layout as the company table (metrics x '2026A' columns).

    'Revenue' is total income and 'EBITDA' is pre-provision operating profit,
    so the generic charts and valuation code still have something sensible to plot.
    """
    inc = financial_data["income_statement"].sort_values("year")
    bal = financial_data["balance_sheet"].set_index("year") if financial_data.get("balance_sheet") is not None else pd.DataFrame()
    rat = financial_data["ratios"].set_index("year") if financial_data.get("ratios") is not None else pd.DataFrame()
    km = financial_data["key_metrics"].set_index("year") if financial_data.get("key_metrics") is not None else pd.DataFrame()

    table = {"metrics": BANK_METRICS_ORDER}
    prev = None
    for _, r in inc.iterrows():
        y = int(r["year"])
        b = bal.loc[y] if y in bal.index else pd.Series(dtype=float)
        rt = rat.loc[y] if y in rat.index else pd.Series(dtype=float)
        k = km.loc[y] if y in km.index else pd.Series(dtype=float)
        nii, other = r.get("netInterestIncome"), r.get("otherIncome")
        total_income = nii + other if pd.notna(nii) and pd.notna(other) else np.nan
        opex, ppop, prov = r.get("operatingExpenses"), r.get("preProvisionOperatingProfit"), r.get("provisions")
        adv, dep = b.get("advances", np.nan), b.get("deposits", np.nan)
        avg_adv = (adv + prev["adv"]) / 2 if prev and pd.notna(prev["adv"]) and pd.notna(adv) else adv

        def growth(cur, key):
            return _pct(_div(cur - prev[key], prev[key])) if prev and pd.notna(prev[key]) and pd.notna(cur) else "N/A"

        col = [
            nii, growth(nii, "nii"), other, total_income, opex,
            _pct(_div(opex, total_income)), ppop, prov, _pct(_div(prov, avg_adv), 2), r.get("netIncome"),
            growth(r.get("netIncome"), "pat"), adv, growth(adv, "adv"), dep, growth(dep, "dep"),
            _pct(_div(adv, dep)), _pct(rt.get("netInterestMargin"), 2), _pct(rt.get("returnOnAssets"), 2),
            _pct(rt.get("returnOnEquity")), _pct(r.get("grossNPARatio"), 2), _pct(r.get("netNPARatio"), 2),
            _pct(r.get("cet1Ratio")), r.get("eps"),
            round(_div(b.get("totalStockholdersEquity", np.nan), r.get("weightedAverageShsOutDil")), 2),
            k.get("peRatio"), k.get("pbRatio"),
            total_income, ppop,
        ]
        table[f"{y}A"] = [None if (isinstance(v, float) and np.isnan(v)) else v for v in col]
        prev = {"nii": nii, "pat": r.get("netIncome"), "adv": adv, "dep": dep}
    return pd.DataFrame(table).astype("object")
