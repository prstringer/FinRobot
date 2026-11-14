#!/usr/bin/env python
# coding: utf-8
"""
Tests for the FinEdge adapter and currency formatting.

They use small synthetic rows shaped like FinEdge responses (no network, no
vendor data). Run from finrobot_equity/core with:  PYTHONPATH=src pytest tests
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules import currency  # noqa: E402
from modules.finedge_api import (  # noqa: E402
    FinEdgeClient,
    FinEdgeError,
    FinEdgePlanError,
    build_bank_metrics_table,
    build_ratios_frames,
    collect_data_warnings,
    is_bank_statement,
    normalize_balance_sheet,
    normalize_cash_flow,
    normalize_income_statement,
    normalize_symbol,
)

CR = 1e7  # one crore


# --- Synthetic FinEdge-shaped rows ------------------------------------------
def company_pl(year, revenue_cr, cogs_cr, employee_cr, other_cr, finance_cr, dna_cr, other_income_cr,
               discontinued_cr=0):
    expenses = cogs_cr + employee_cr + other_cr + finance_cr + dna_cr
    pbt = revenue_cr + other_income_cr - expenses
    return {
        "year": year, "period_start": int(f"{year - 1}0401"), "period_end": int(f"{year}0331"),
        "result_date": int(f"{year}0520"),
        "revenueFromOperations": int(revenue_cr * CR), "income": int((revenue_cr + other_income_cr) * CR),
        "otherIncome": int(other_income_cr * CR), "costofGoodsSold": int(cogs_cr * CR),
        "costOfMaterialsConsumed": int(cogs_cr * CR), "purchasesOfStockInTrade": 0, "changesInInventories": 0,
        "employeeBenefitExpense": int(employee_cr * CR), "otherExpenses": int(other_cr * CR),
        "financeCosts": int(finance_cr * CR), "depreciationAndAmortisation": int(dna_cr * CR),
        "expenses": int(expenses * CR), "exceptionalItemsBeforeTax": 0,
        "profitBeforeTax": int(pbt * CR), "taxExpense": int(pbt * 0.25 * CR),
        "profitLossForPeriod": int(pbt * 0.75 * CR), "profitOrLossAttributableToOwners": int(pbt * 0.75 * CR),
        "profitLossFromDiscontinuedOperationsAfterTax": int(discontinued_cr * CR),
        "eps": round(pbt * 0.75 / 100, 2), "dilutedOutstandingShares": int(100 * CR),
        # Bank-only schema lines that FinEdge returns as 0 for companies
        "interestEarned": 0,
    }


def company_bs(year, debt_cr, cash_cr, equity_cr):
    return {
        "year": year, "period_end": int(f"{year}0331"),
        "assets": int((equity_cr + debt_cr + 100) * CR), "liabilities": int((debt_cr + 100) * CR),
        "borrowingsCurrent": int(debt_cr * 0.5 * CR), "borrowingsNoncurrent": int(debt_cr * 0.5 * CR),
        "cashAndCashEquivalents": int(cash_cr * CR), "otherBankBalances": 0, "currentInvestments": 0,
        "equityAttributableToOwnersOfParent": int(equity_cr * CR), "totalEquity": int(equity_cr * CR),
        "currentAssets": int(500 * CR), "currentLiabilities": int(250 * CR),
        # ITC-style schema: bank lines present but zero
        "deposits": 0, "advances": 0,
    }


def company_cf(year, ocf_cr, capex_cr):
    return {"year": year, "period_end": int(f"{year}0331"),
            "cashFlowsFromOperatingActivities": int(ocf_cr * CR),
            "purchaseOfFixed&IntangibleAssets": -int(capex_cr * CR),
            "cashFlowsFromInvestingActivities": -int(capex_cr * CR),
            "cashFlowsFromFinancingActivities": -int(50 * CR), "netCashFlow": int(10 * CR)}


def bank_pl(year, int_earned_cr, int_exp_cr, other_cr, employees_cr, opex_cr, prov_cr, gnpa=0.012, cet1=0.19):
    income = int_earned_cr + other_cr
    expenditure = int_exp_cr + employees_cr + opex_cr
    pbt = income - expenditure - prov_cr
    return {
        "year": year, "period_end": int(f"{year}0331"),
        "interestEarned": int(int_earned_cr * CR), "interestExpended": int(int_exp_cr * CR),
        "otherIncome": int(other_cr * CR), "income": int(income * CR),
        "employeesCost": int(employees_cr * CR), "otherOperatingExpenses": int(opex_cr * CR),
        "expenditureExcludingProvisions": int(expenditure * CR), "provisionsForLoanLoss": int(prov_cr * CR),
        "profitLossBeforeTax": int(pbt * CR), "taxExpense": int(pbt * 0.25 * CR),
        "profitLossForThePeriod": int(pbt * 0.75 * CR), "eps": round(pbt * 0.75 / 100, 2),
        "dilutedOutstandingShares": int(100 * CR), "grossNonPerformingAssets": int(10 * CR),
        "nonPerformingAssets": int(3 * CR), "percentageOfGrossNpa": gnpa, "percentageOfNpa": gnpa / 3,
        "cET1Ratio": cet1, "additionalTier1Ratio": 0,
    }


def bank_bs(year, advances_cr, deposits_cr, equity_cr):
    return {"year": year, "period_end": int(f"{year}0331"), "advances": int(advances_cr * CR),
            "deposits": int(deposits_cr * CR), "investments": int(300 * CR), "borrowings": int(100 * CR),
            "capital": int(10 * CR), "reserves": int((equity_cr - 10) * CR),
            "assets": int((advances_cr + 400) * CR), "cashAndBalancesWithRBI": int(50 * CR),
            "bankBalancesAndMoneyAtCallAndShortNotice": int(20 * CR), "goodwillOnConsolidation": 0}


# --- Tests ----------------------------------------------------------------------
def test_normalize_symbol():
    assert normalize_symbol("itc.ns") == "ITC"
    assert normalize_symbol("RELIANCE.BO") == "RELIANCE"
    assert normalize_symbol(" HDFCBANK ") == "HDFCBANK"


def test_company_income_statement_derives_ebitda_and_sga():
    rows = [company_pl(2026, 1000, 400, 100, 150, 10, 40, 30), company_pl(2025, 900, 360, 90, 140, 10, 35, 25)]
    df = normalize_income_statement(rows)
    latest = df.iloc[0]  # newest first
    assert latest["year"] == 2026
    assert latest["revenue"] == pytest.approx(1000 * CR)
    assert latest["grossProfit"] == pytest.approx(600 * CR)
    assert latest["sellingGeneralAndAdministrativeExpenses"] == pytest.approx(250 * CR)
    # EBITDA = revenue - (expenses - finance - D&A) = 1000 - 650 = 350 (excludes other income)
    assert latest["ebitda"] == pytest.approx(350 * CR)
    assert latest["operatingIncome"] == pytest.approx(310 * CR)
    # Gross profit - SG&A must equal EBITDA, as the report's tables assume
    assert latest["grossProfit"] - latest["sellingGeneralAndAdministrativeExpenses"] == pytest.approx(latest["ebitda"])


def test_zero_deposits_do_not_trigger_bank_layout():
    """Company balance sheets carry deposits/advances as 0; they must not be read as banks."""
    pl = [company_pl(2026, 1000, 400, 100, 150, 10, 40, 30)]
    bs = [company_bs(2026, debt_cr=200, cash_cr=50, equity_cr=800)]
    assert not is_bank_statement(pl, bs)
    df = normalize_balance_sheet(bs)
    assert df.iloc[0]["totalDebt"] == pytest.approx(200 * CR)
    assert df.iloc[0]["netDebt"] == pytest.approx(150 * CR)
    assert df.iloc[0]["totalStockholdersEquity"] == pytest.approx(800 * CR)


def test_cash_flow_capex_sign_and_fcf():
    df = normalize_cash_flow([company_cf(2026, ocf_cr=300, capex_cr=80)])
    assert df.iloc[0]["capitalExpenditure"] == pytest.approx(-80 * CR)
    assert df.iloc[0]["freeCashFlow"] == pytest.approx(220 * CR)


def test_ratios_and_ev_ebitda_use_annual_rows_and_treat_zero_as_missing():
    years = [2026, 2025]
    inc = normalize_income_statement([company_pl(y, 1000, 400, 100, 150, 10, 40, 30) for y in years])
    bal = normalize_balance_sheet([company_bs(y, 200, 50, 800) for y in years])
    cfs = normalize_cash_flow([company_cf(y, 300, 80) for y in years])
    ratio_rows = {
        "pr": [{"header": "TTM", "year": 2026, "returnOnEquity": 0.3},
               {"header": "Mar 2026", "year": 2026, "returnOnEquity": 0.25},
               {"header": "Mar 2025", "year": 2025, "returnOnEquity": 0.24}],
        "le": [{"header": "Mar 2026", "year": 2026, "totalDebtToEquity": 0.25}],
        "li": [{"header": "Mar 2026", "year": 2026, "interestCoverage": 35.0, "currentRatio": 2.0}],
        "ef": [],
    }
    price_rows = [
        {"header": "TTM", "year": 2026, "average_price": 99, "pe": 30, "pb": 5},
        {"header": "Mar 2026", "year": 2026, "average_price": 50, "pe": 20, "pb": 4},
        {"header": "Dec 2025", "year": 2025, "average_price": 45, "pe": 18, "pb": 0},   # interim row, PB=0
        {"header": "Mar 2025", "year": 2025, "average_price": 40, "pe": 16, "pb": 0},   # PB=0 means missing
    ]
    ratios, km = build_ratios_frames(ratio_rows, price_rows, inc, bal, cfs)
    r26 = ratios.set_index("year").loc[2026]
    assert r26["returnOnEquity"] == pytest.approx(0.25)      # not the TTM row
    assert r26["priceEarningsRatio"] == pytest.approx(20)
    assert r26["cashFlowToDebtRatio"] == pytest.approx(1.5)  # 300 / 200
    assert np.isnan(ratios.set_index("year").loc[2025, "priceToBookRatio"])
    k26 = km.set_index("year").loc[2026]
    # market cap = 50 x 100 crore shares; EV = mcap + debt 200 cr - cash 50 cr; EBITDA 350 cr
    expected_ev = 50 * 100 * CR + 200 * CR - 50 * CR
    assert k26["enterpriseValue"] == pytest.approx(expected_ev)
    assert k26["enterpriseValueOverEBITDA"] == pytest.approx(expected_ev / (350 * CR))
    for col in ["calendarYear", "debtEquityRatio", "debtRatio", "interestCoverage", "netProfitMargin",
                "currentRatio", "cashFlowToDebtRatio"]:
        assert col in ratios.columns  # columns the report's credit table needs


def test_bank_layout_and_metrics_table():
    pl = [bank_pl(2026, 300, 180, 60, 25, 45, 20), bank_pl(2025, 280, 170, 50, 24, 40, 12)]
    bs = [bank_bs(2026, 2900, 3100, 470), bank_bs(2025, 2600, 2700, 420)]
    assert is_bank_statement(pl, bs)
    inc, bal = normalize_income_statement(pl), normalize_balance_sheet(bs)
    assert inc.iloc[0]["netInterestIncome"] == pytest.approx(120 * CR)
    assert inc.iloc[0]["preProvisionOperatingProfit"] == pytest.approx(110 * CR)  # 360 - 250
    assert np.isnan(inc.iloc[0]["ebitda"])
    ratios = pd.DataFrame([{"year": 2026, "netInterestMargin": 0.031, "returnOnAssets": 0.017,
                            "returnOnEquity": 0.134}])
    km = pd.DataFrame([{"year": 2026, "peRatio": 19.7, "pbRatio": 2.6}])
    table = build_bank_metrics_table({"income_statement": inc, "balance_sheet": bal,
                                      "ratios": ratios, "key_metrics": km}).set_index("metrics")
    assert table.loc["Net Interest Income", "2026A"] == pytest.approx(120 * CR)
    assert table.loc["NII Growth", "2026A"] == "9.1%"           # 120 / 110 - 1
    assert table.loc["Cost-to-Income", "2026A"] == "38.9%"      # 70 / 180
    assert table.loc["Credit-Deposit Ratio", "2026A"] == "93.5%"
    assert table.loc["NIM", "2026A"] == "3.10%"
    assert table.loc["Gross NPA", "2026A"] == "1.20%"
    assert table.loc["Book Value per Share", "2026A"] == pytest.approx(4.7)
    # Generic charts still get Revenue (= total income) and EBITDA (= pre-provision profit)
    assert table.loc["Revenue", "2026A"] == pytest.approx(180 * CR)
    assert table.loc["EBITDA", "2026A"] == pytest.approx(110 * CR)


def test_data_warnings_flag_one_offs_and_merger_jumps():
    inc = normalize_income_statement([company_pl(2025, 1000, 400, 100, 150, 10, 40, 30, discontinued_cr=150)])
    w = collect_data_warnings("XYZ", inc, pd.DataFrame(), bank=False, limit=1)
    assert any("discontinued operations" in x and "FY2025" in x for x in w)
    bal = normalize_balance_sheet([bank_bs(2024, 2400, 2300, 400), bank_bs(2023, 1600, 1800, 300)])
    w = collect_data_warnings("BANK", pd.DataFrame(), pd.DataFrame(), bank=True, limit=5, balance_df=bal)
    assert any("Advances grew 50% in FY2024" in x for x in w)


class _FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def test_client_plan_error_and_token_masking():
    token = "secret-token-123"
    session = _FakeSession([_FakeResponse(401, text=f"ER0012 - user not authorized ({token})")])
    client = FinEdgeClient(token=token, session=session, pause=0)
    with pytest.raises(FinEdgePlanError) as exc:
        client.profile("TCS")
    assert token not in str(exc.value)
    assert session.calls[0][1]["token"] == token  # auth goes in the query string


def test_client_caches_and_reports_bad_requests():
    session = _FakeSession([_FakeResponse(200, {"peers": ["A", "B"]}),
                            _FakeResponse(400, text="ratio type selection pr is emtpy or invalid")])
    client = FinEdgeClient(token="t", session=session, pause=0)
    assert client.peers("ITC") == ["A", "B"]
    assert client.peers("ITC") == ["A", "B"]  # cached, no second HTTP call
    assert len(session.calls) == 1
    with pytest.raises(FinEdgeError):
        client.get("/api/v1/financial-metrics/ITC", ratio_type="pr")


def test_client_requires_token(monkeypatch):
    monkeypatch.delenv("FINEDGE_API_TOKEN", raising=False)
    with pytest.raises(FinEdgeError):
        FinEdgeClient(token=None)


def test_currency_formatting_inr_and_usd_unchanged():
    try:
        currency.set_currency("USD")
        assert currency.fmt_amount(899.1e9) == "$899.1B"
        assert currency.fmt_price(12.5) == "$12.50"
        currency.set_currency("INR")
        assert currency.fmt_amount(899133000000) == "₹89,913 Cr"
        assert currency.fmt_price(284.05) == "₹284.05"
        assert currency.fmt_market_cap(355929.3) == "₹355,929 Cr"
        assert currency.parse_money("₹1,339.60") == pytest.approx(1339.6)
        assert currency.chart_scale() == (1e7, "₹ Cr")
    finally:
        currency.set_currency("USD")


def test_html_value_formatting_follows_currency():
    from modules.html_renderer import _format_value
    try:
        currency.set_currency("INR")
        assert _format_value(899133000000.0) == "₹89,913 Cr"
        assert _format_value("12.5%") == "12.5%"
        currency.set_currency("USD")
        assert _format_value(899133000000.0) == "$899.1B"
    finally:
        currency.set_currency("USD")


def test_forecasts_tolerate_bank_table_without_margin_rows():
    from modules.financial_data_processor import calculate_growth_and_forecasts
    df = pd.DataFrame({"metrics": ["Revenue", "EBITDA", "EPS", "PE Ratio"],
                       "2025A": [100.0, 50.0, 10.0, 20.0], "2026A": [110.0, 55.0, 11.0, 19.0]}).astype("object")
    out = calculate_growth_and_forecasts(df, {
        "revenue_base_year": "2026A",
        "revenue_growth_assumptions": {"2027E": 0.1, "2028E": 0.1, "2029E": 0.1},
        "margin_improvement": {}, "sga_margin_change": 0})
    assert out.loc[out["metrics"] == "Revenue", "2027E"].iloc[0] == pytest.approx(121.0)
    assert out.loc[out["metrics"] == "EPS", "2027E"].iloc[0] == pytest.approx(12.1)


def test_multiple_valuation_for_banks():
    from modules.valuation_engine import ValuationEngine
    engine = ValuationEngine({"analysis": pd.DataFrame(), "current_price": 800, "shares_outstanding": 1})
    r = engine.calculate_multiple_valuation("P/B Multiple", 360.0, [3.0, 2.5, 2.5, 0, None], confidence=0.7)
    assert r.target_price == pytest.approx(360.0 * (8.0 / 3))  # zeros/None ignored
    assert r.low_estimate < r.target_price < r.high_estimate
    skipped = engine.calculate_multiple_valuation("P/E Multiple", None, [20.0])
    assert skipped.target_price == 0


def test_pdf_symbol_override_and_safe_paragraph():
    from modules.professional_pdf_report import Paragraph, _pdf_safe
    try:
        currency.set_currency("INR")
        currency.set_symbol_override("Rs ")
        assert currency.fmt_price(284.05) == "Rs 284.05"
        assert currency.fmt_amount(899133000000) == "Rs 89,913 Cr"
        assert currency.chart_scale() == (1e7, "Rs Cr")
        assert currency.parse_money("Rs 1,339.60") == pytest.approx(1339.6)
        assert _pdf_safe("Revenue ₹89,913 Cr") == "Revenue Rs 89,913 Cr"
        assert "₹" not in Paragraph("₹10").text
        currency.set_currency("INR")  # resets the override
        assert currency.fmt_price(1) == "₹1.00"
    finally:
        currency.set_currency("USD")
