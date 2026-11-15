"""Tests for the canonical report context: forecast bridge, units, peer median, current valuation, consistency checks."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules import report_context as rc  # noqa: E402
from modules.financial_data_processor import build_forecast_pnl_bridge, calculate_growth_and_forecasts  # noqa: E402
from modules.finedge_api import get_finedge_ttm_and_latest_balance  # noqa: E402
from modules.market_data_api import compute_technical_indicators  # noqa: E402

CR = 1e7


def _income():
    return pd.DataFrame([
        {"year": 2026, "date": pd.Timestamp("2026-03-31"), "revenue": 93991.1 * 1e6, "ebitda": 7932.1e6,
         "depreciationAndAmortization": 3940.9e6, "interestExpense": 1614.8e6, "otherIncome": 688.0e6,
         "incomeBeforeTax": 2874.7e6, "incomeTaxExpense": 745.8e6, "netIncome": 2044.4e6,
         "netIncomeIncludingMinority": 2128.9e6, "eps": 2.13, "weightedAverageShsOutDil": 959152514.0,
         "costOfRevenue": 50531.7e6, "operatingIncome": 3991.2e6},
        {"year": 2025, "date": pd.Timestamp("2025-03-31"), "revenue": 105450.9e6, "ebitda": 12985.0e6,
         "depreciationAndAmortization": 3733.9e6, "interestExpense": 2174.7e6, "otherIncome": 1521.5e6,
         "incomeBeforeTax": 8597.9e6, "incomeTaxExpense": 2161.6e6, "netIncome": 6391.6e6,
         "netIncomeIncludingMinority": 6440.2e6, "eps": 6.66, "weightedAverageShsOutDil": 959152514.0,
         "costOfRevenue": 56755.4e6, "operatingIncome": 9251.1e6},
    ])


def _final_df():
    """Analysis table in absolute rupees for 2025A/2026A (what the CSV holds)."""
    rows = {
        "Revenue": [105450.9e6, 93991.1e6], "Cost of Operations": [56755.4e6, 50531.7e6],
        "SG&A": [35710.5e6, 35527.3e6], "Contribution Profit": [48695.5e6, 43459.4e6],
        "Contribution Margin": ["46.2%", "46.2%"], "EBITDA": [12985.0e6, 7932.1e6],
        "EBITDA Margin": ["12.3%", "8.4%"], "SG&A Margin": ["33.9%", "37.8%"],
        "Revenue Growth": ["8.9%", "-10.9%"], "EPS": [6.66, 2.13], "PE Ratio": [23.07, 60.61],
    }
    df = pd.DataFrame({"metrics": list(rows), "2025A": [v[0] for v in rows.values()], "2026A": [v[1] for v in rows.values()]})
    return df.astype("object")


CONFIG = {"revenue_base_year": "2026A",
          "revenue_growth_assumptions": {"2027E": 0.05, "2028E": 0.06, "2029E": 0.04},
          "margin_improvement": {"Contribution Margin": 0.01, "EBITDA Margin": 0.01}, "sga_margin_change": -0.005}


@pytest.fixture(scope="module")
def forecast():
    return calculate_growth_and_forecasts(_final_df(), CONFIG, income_df=_income(), current_price=249.0)


def test_forecast_eps_is_rebuilt_from_ebitda_not_grown_off_base(forecast):
    eps = {y: float(forecast.loc[forecast["metrics"] == "EPS", y].iloc[0]) for y in ("2026A", "2027E", "2028E", "2029E")}
    old_method_fy29 = 2.13 * ((1.05 * 1.01) * (1.06 * 1.01) * (1.04 * 1.01))
    assert eps["2029E"] > 2 * old_method_fy29 * 0.9          # no longer ~2.5
    assert eps["2027E"] > eps["2026A"] and eps["2029E"] > eps["2028E"] > eps["2027E"]
    # FY29E EBITDA is higher than FY25A, so EPS must not be ~60% lower than FY25A
    assert eps["2029E"] > 0.8 * 6.66 * 0.9


def test_bridge_reconciles_line_by_line(forecast):
    bridge = forecast.attrs["forecast_bridge"]
    a = forecast.attrs["forecast_bridge_assumptions"]
    for y, r in bridge.iterrows():
        pbt = r["ebitda"] - r["depreciation"] - r["interest"] + r["other_income"]
        assert r["pretax_income"] == pytest.approx(pbt)
        assert r["pat_total"] == pytest.approx(r["pretax_income"] - r["tax"])
        assert r["tax"] == pytest.approx(r["pretax_income"] * a["tax_rate"])
        assert r["eps"] * r["shares"] == pytest.approx(r["pat_owners"])
    assert a["tax_rate"] == pytest.approx(745.8 / 2874.7)


def test_forecast_pe_uses_current_price_over_bridge_eps(forecast):
    pe_row = forecast[forecast["metrics"].astype(str).str.startswith("PE Ratio")]
    assert len(pe_row) == 1
    for y in ("2027E", "2028E", "2029E"):
        eps = float(forecast.loc[forecast["metrics"] == "EPS", y].iloc[0])
        assert float(pe_row[y].iloc[0]) == pytest.approx(249.0 / eps, abs=0.06)
    assert "FY-avg" in pe_row["metrics"].iloc[0]          # the basis is part of the row label


def test_bridge_unavailable_without_inputs():
    assert build_forecast_pnl_bridge(_final_df(), _income().drop(columns=["interestExpense"]), ["2027E"], "2026A") is None


def test_scale_table_to_crore_is_single_canonical_unit():
    t = rc.scale_table_to_crore(_final_df())
    rev = t[t["metrics"] == "Revenue (INR crore)"]["2026A"].iloc[0]
    assert rev == 9399.1                                   # not 93,991
    assert t[t["metrics"] == "EPS"]["2026A"].iloc[0] == 2.13   # per-share values are not rescaled
    assert t[t["metrics"] == "EBITDA Margin"]["2026A"].iloc[0] == "8.4%"


def test_peer_median_excludes_subject_and_is_computed_in_code():
    ev = pd.DataFrame({"WELSPUNLIV": [16.59], "KPRMILL": [26.95], "VTL": [11.36], "TRIDENT": [17.79],
                       "ICIL": [15.99], "GARFIBRES": [25.66]}, index=[2026])
    ctx = _ctx(peer_ev_df=ev)
    p = ctx["peers"]["ev_ebitda"]
    assert p["median_excl_subject"] == pytest.approx(17.79)
    assert p["median_incl_subject"] == pytest.approx(17.19)
    assert p["median_peer_name"] == "TRIDENT"
    assert "EXCLUDES" in p["policy"]


def _ctx(peer_ev_df=None, price=249.0, mcap_cr=23524.34, net_debt=-15.5e6):
    final = calculate_growth_and_forecasts(_final_df(), CONFIG, income_df=_income(), current_price=price)
    balance = pd.DataFrame([{"year": 2026, "date": pd.Timestamp("2026-03-31"), "totalDebt": 18023.7e6,
                             "cashAndShortTermInvestments": 10246.0e6, "netDebt": 7777.7e6,
                             "totalStockholdersEquity": 49171.9e6, "minorityInterest": 607.6e6,
                             "inventory": 21706.9e6, "netReceivables": 13122.4e6, "accountPayables": 13766.9e6}])
    km = pd.DataFrame([{"year": 2026, "averagePrice": 129.11, "marketCap": 123836.18e6, "netDebt": 7777.7e6,
                        "enterpriseValue": 131613.88e6, "enterpriseValueOverEBITDA": 16.59, "peRatio": 60.61,
                        "netIncomePerShare": 2.13, "shares": 959152514.0}])
    cash = pd.DataFrame([{"year": 2026, "capitalExpenditure": -7986e6, "operatingCashFlow": 11749.8e6, "freeCashFlow": 3763.8e6}])
    ttm = {"ttm": {"period_end": "2026-09-30", "quarters": ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"],
                   "revenue": 103310.9e6, "ebitda": 10684.7e6, "depreciation": 4085.7e6, "interest": 1439.3e6,
                   "pretax_income": 6347.0e6, "net_income": 4337.4e6, "eps": 4.57, "shares": 944752514.0},
           "latest_balance": {"date": "2026-09-30", "total_debt": 19046.4e6, "cash_and_st_investments": 19061.9e6,
                              "net_debt": net_debt, "equity": 50660.2e6, "minority_interest": 700e6}}
    cm = {"share_price": price, "market_cap": mcap_cr, "fwd_pe": 54.49, "pb_ratio": 4.71, "roe": 8.56,
          "dividend_yield": 0.04, "52w_range": (107.4, 267.4)}
    return rc.build_report_context(
        ticker="WELSPUNLIV", company_name="Welspun Living Ltd", as_of="2026-11-15", final_df=final,
        financial_data={"income_statement": _income(), "balance_sheet": balance, "cash_flow": cash, "key_metrics": km},
        company_metrics=cm, ttm_balance=ttm, peer_ev_df=peer_ev_df,
        technicals={"price": 249.0, "sma20": 232.76, "sma50": 224.38, "sma200": 163.43, "rsi14": 67.8,
                    "macd": 9.36, "macd_signal": 8.11, "macd_histogram": 1.24})


def test_current_valuation_is_at_current_price_not_fy_average():
    cur = _ctx()["current_valuation"]
    assert cur["ev_cr"] == pytest.approx(23524.34 - 1.55)                 # market cap + latest net debt
    assert cur["ev_ebitda_latest_fy"] == pytest.approx(23522.8 / 793.21, rel=1e-3)   # ~29.7x, not 16.6x
    assert cur["ev_ebitda_ttm"] == pytest.approx(23522.8 / 1068.47, rel=1e-3)        # ~22.0x
    assert cur["pe_latest_fy_at_current_price"] == pytest.approx(249 / 2.13)         # ~116.9x
    assert cur["pe_ttm_api"] == 54.49


def test_fair_value_grid_compares_with_current_ev():
    fv = _ctx()["fair_value"]
    g14 = next(g for g in fv["grid"] if g["multiple"] == 14)
    assert g14["ev_cr"] == pytest.approx(14 * 977.04, rel=1e-3)
    assert g14["vs_current_ev_pct"] == pytest.approx((14 * 977.04 / 23522.8 - 1) * 100, rel=1e-2)
    assert g14["vs_current_ev_pct"] < -40                                 # a 4-19% "upside" is not what the data says


def test_model_forecast_and_peer_projection_are_separate_labelled_keys():
    ev = pd.DataFrame({"WELSPUNLIV": [16.59], "TRIDENT": [17.79]}, index=[2026])
    actual = pd.DataFrame({"WELSPUNLIV": [7932.1e6], "TRIDENT": [8691.8e6]}, index=[2026])
    proj = actual.copy()
    proj.loc[2027] = [7797.9e6, 7696.9e6]
    ctx = rc.build_report_context(**{**_ctx_kwargs(), "peer_ev_df": ev, "peer_ebitda_actual": actual, "peer_ebitda_projected": proj})
    assert ctx["model_forecast"]["2027E"]["ebitda_cr"] == pytest.approx(977.04, rel=1e-3)
    tp = ctx["peers"]["trend_projection"]
    assert tp["ebitda_cr"]["WELSPUNLIV"] == pytest.approx(779.8, rel=1e-3)
    assert tp["is_consensus"] is False and "NOT" in tp["note"]
    assert "NOT analyst consensus" in ctx["model_forecast_note"]
    block = ctx["prompt_block"]
    assert "MODEL FORECAST (our own projection, NOT analyst consensus" in block
    assert "PEER-DATASET TREND PROJECTION" in block and "NOT consensus" in block


def _ctx_kwargs():
    return dict(ticker="WELSPUNLIV", company_name="Welspun Living Ltd", as_of="2026-11-15",
                final_df=calculate_growth_and_forecasts(_final_df(), CONFIG, income_df=_income(), current_price=249.0),
                financial_data=_FD, company_metrics=_CM, ttm_balance=_TTM, technicals={"price": 249.0, "sma20": 232.76, "rsi14": 67.8})


# fixtures reused by _ctx_kwargs
_FD = {"income_statement": _income(),
       "balance_sheet": pd.DataFrame([{"year": 2026, "date": pd.Timestamp("2026-03-31"), "totalDebt": 18023.7e6,
                                       "cashAndShortTermInvestments": 10246.0e6, "netDebt": 7777.7e6,
                                       "totalStockholdersEquity": 49171.9e6, "minorityInterest": 607.6e6,
                                       "inventory": 21706.9e6, "netReceivables": 13122.4e6, "accountPayables": 13766.9e6}]),
       "cash_flow": pd.DataFrame([{"year": 2026, "capitalExpenditure": -7986e6, "operatingCashFlow": 11749.8e6, "freeCashFlow": 3763.8e6}]),
       "key_metrics": pd.DataFrame([{"year": 2026, "averagePrice": 129.11, "marketCap": 123836.18e6, "netDebt": 7777.7e6,
                                     "enterpriseValue": 131613.88e6, "enterpriseValueOverEBITDA": 16.59, "peRatio": 60.61,
                                     "netIncomePerShare": 2.13, "shares": 959152514.0}])}
_CM = {"share_price": 249.0, "market_cap": 23524.34, "fwd_pe": 54.49, "pb_ratio": 4.71, "roe": 8.56,
       "dividend_yield": 0.04, "52w_range": (107.4, 267.4)}
_TTM = {"ttm": {"period_end": "2026-09-30", "quarters": ["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"],
                "revenue": 103310.9e6, "ebitda": 10684.7e6, "depreciation": 4085.7e6, "interest": 1439.3e6,
                "pretax_income": 6347.0e6, "net_income": 4337.4e6, "eps": 4.57, "shares": 944752514.0},
        "latest_balance": {"date": "2026-09-30", "total_debt": 19046.4e6, "cash_and_st_investments": 19061.9e6,
                           "net_debt": -15.5e6, "equity": 50660.2e6, "minority_interest": 700e6}}


def test_all_consistency_checks_pass_on_consistent_inputs():
    ctx = _ctx()
    assert rc.failed_checks(ctx) == [], rc.failed_checks(ctx)
    names = [c["name"] for c in ctx["checks"]]
    for needed in ("price x shares = market cap (Cr)", "EPS x shares = PAT (TTM)", "P/E latest FY at current price = price / EPS"):
        assert any(needed in n for n in names)


def test_checks_catch_an_inconsistent_market_cap():
    ctx = _ctx(mcap_cr=23524.34)
    ctx["market"]["market_cap_cr"] = 12000.0
    failed = [c["name"] for c in rc.run_checks(ctx) if c["ok"] is False]
    assert "price x shares = market cap (Cr)" in failed


def test_checks_catch_pe_on_the_wrong_price():
    ctx = _ctx()
    ctx["fy_series"]["2026A"]["pe"] = 117.0     # a current-price P/E mislabelled as FY-avg-price
    failed = [c["name"] for c in rc.run_checks(ctx) if c["ok"] is False]
    assert "P/E 2026A = FY-avg price / EPS" in failed


def test_section_contexts_identical_and_detects_drift():
    ctx = _ctx()
    from modules.text_generator_agents import _prepare_user_prompt
    data = rc.build_text_data(ctx, _final_df(), None, None, base={"data_notes": []})
    prompts = {t: _prepare_user_prompt(data, t, "Welspun Living Ltd", "WELSPUNLIV")
               for t in ("tagline", "valuation_overview", "news_summary")}
    rc.assert_section_contexts_identical(prompts)
    assert "93,991" not in prompts["news_summary"] and "9399.1" in prompts["news_summary"]
    prompts["news_summary"] = prompts["news_summary"].replace("9399.1", "93991.0")
    with pytest.raises(AssertionError):
        rc.assert_section_contexts_identical(prompts)


def test_narrative_lint_flags_tenfold_unit_slip():
    ctx = _ctx()
    bad = "FY26 revenue fell to roughly 93,991 crore and EBITDA to about 7,932 crore."
    good = "FY26 revenue fell to roughly 9,399 crore and EBITDA to about 793 crore."
    assert len(rc.lint_narrative_units(bad, ctx)) == 2
    assert rc.lint_narrative_units(good, ctx) == []


def test_roe_and_pe_bases_are_labelled_and_distinct():
    r = _ctx()["ratios"]
    assert r["roe_ttm_pct"] == 8.56 and r["roe_latest_fy_pct"] == pytest.approx(2044.4 / 49171.9 * 100, rel=1e-3)
    assert "TTM" in r["roe_ttm_basis"] and "FY" in r["roe_latest_fy_basis"]
    assert r["roe_ttm_pct_recomputed"] == pytest.approx(4337.4 / 50660.2 * 100, rel=1e-3)


def test_derived_fields_present_and_unavailable_inputs_logged():
    d = _ctx()["derived"]
    assert d["fcf_cr"]["2026A"] == pytest.approx(376.38, rel=1e-3)
    assert d["capex_cr"]["2026A"] == pytest.approx(-798.6, rel=1e-3)
    assert d["interest_coverage_ebit_x"]["2026A"] == pytest.approx(3991.2 / 1614.8, rel=1e-3)
    assert d["working_capital_days"]["2026A"]["inventory_days"] == pytest.approx(21706.9 / 50531.7 * 365, rel=1e-3)
    assert d["shares_outstanding_mn"] == pytest.approx(944.75, rel=1e-3)


def test_technicals_exported_as_named_values():
    t = _ctx()["technicals"]
    for k in ("rsi14", "macd", "macd_signal", "macd_histogram", "sma20", "sma50", "sma200", "week52_low", "week52_high",
              "price_vs_sma20_pct", "price_vs_sma50_pct", "price_vs_sma200_pct", "pct_below_52w_high"):
        assert t.get(k) is not None, k
    assert "RSI14 67.8" in _ctx()["prompt_block"] and "52-week range" in _ctx()["prompt_block"]


def test_compute_technical_indicators_adds_sma20_and_52w_range():
    n = 260
    close = pd.Series(np.linspace(100, 160, n))
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n), "close": close, "high": close + 2,
                       "low": close - 2, "volume": 1000.0})
    out = compute_technical_indicators(df, "X")
    assert out["sma20"] == pytest.approx(close.tail(20).mean(), abs=0.01)
    assert out["high_52w"] == pytest.approx(close.tail(250).max() + 2, abs=0.01)
    assert out["low_52w"] == pytest.approx(close.tail(250).min() - 2, abs=0.01)


class _StubClient:
    def __init__(self, quarters):
        self.quarters = quarters

    def financials(self, symbol, code, period, st):
        if code == "pl" and period == "quarterly":
            return self.quarters
        return []


def _q(end, rev, eps, ni):
    return {"period_end": end, "revenueFromOperations": rev, "expenses": rev * 0.88, "financeCosts": 1e8,
            "depreciationAndAmortisation": 1e8, "profitBeforeTax": ni * 1.3, "taxExpense": ni * 0.3,
            "profitOrLossAttributableToOwners": ni, "eps": eps, "dilutedOutstandingShares": 9.5e8, "year": int(str(end)[:4])}


def test_ttm_requires_four_consecutive_quarters():
    good = [_q(20260930, 3e10, 1.8, 1.7e9), _q(20260630, 3e10, 1.7, 1.6e9), _q(20260331, 2.5e10, 1.1, 1.0e9), _q(20251231, 2.3e10, 0, 2e6)]
    out = get_finedge_ttm_and_latest_balance("X", _StubClient(good))
    assert out["ttm"] is not None and out["ttm"]["period_end"] == "2026-09-30"
    assert out["ttm"]["revenue"] == pytest.approx(3e10 + 3e10 + 2.5e10 + 2.3e10)
    gap = [good[0], good[1], good[2], _q(20250630, 2.3e10, 0.9, 8e8)]       # Dec-2025 quarter missing
    assert get_finedge_ttm_and_latest_balance("X", _StubClient(gap))["ttm"] is None


def test_reconciliation_log_lists_basis_for_every_multiple(tmp_path):
    ctx = _ctx(peer_ev_df=pd.DataFrame({"WELSPUNLIV": [16.59], "TRIDENT": [17.79], "VTL": [11.36]}, index=[2026]))
    path = tmp_path / "log.md"
    rc.write_reconciliation_log(ctx, str(path))
    text = path.read_text()
    assert "FY AVERAGE" in text and "(current" in text and "PASS" in text
    assert "EV/EBITDA, latest FY, CURRENT" in text and "TTM EBITDA" in text
    assert "median excl. subject" in text and "EXCLUDES" in text


def test_current_valuation_rows_shared_table():
    rows = rc.current_valuation_rows(_ctx())
    labels = [r[0] for r in rows]
    assert "Enterprise value" in labels and "EV/EBITDA (TTM)" in labels
    assert any(l.startswith("Memo:") for l in labels)       # the FY-avg history is labelled as a memo, not mixed in
