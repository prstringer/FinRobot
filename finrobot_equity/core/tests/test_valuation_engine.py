#!/usr/bin/env python
# coding: utf-8
"""Valuation engine: which year's figures are used, net debt and multiples.

Run from finrobot_equity/core with:  PYTHONPATH=src pytest tests
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.valuation_engine import ValuationEngine  # noqa: E402


def analysis_table():
    # Actual years followed by forecasts, as generate_financial_analysis.py writes them
    return pd.DataFrame({
        "metrics": ["Revenue", "EBITDA", "EBITDA Margin"],
        "2024A": [1000.0, 200.0, "20.0%"],
        "2025A": [1100.0, 230.0, "20.9%"],
        "2026E": [1200.0, 400.0, "33.3%"],
        "2027E": [1300.0, 500.0, "38.5%"],
        "CAGR": ["5%", "N/A", "N/A"],
    }).astype("object")


def engine(**extra):
    data = {"analysis": analysis_table(), "current_price": 10.0, "shares_outstanding": 100.0}
    data.update(extra)
    return ValuationEngine(data)


def test_get_metric_defaults_to_latest_actual_year():
    e = engine()
    assert e._get_metric_with_year("EBITDA") == (230.0, "2025A")
    assert e._get_metric("EBITDA") == 230.0


def test_get_metric_can_include_estimates_or_name_a_year():
    e = engine()
    assert e._get_metric_with_year("EBITDA", include_estimates=True) == (500.0, "2027E")
    assert e._get_metric("EBITDA", year="2024A") == 200.0
    assert e._get_metric("Missing Metric") is None


def test_get_metric_skips_years_without_numbers():
    df = analysis_table()
    df.loc[df["metrics"] == "EBITDA", "2025A"] = None
    e = ValuationEngine({"analysis": df, "current_price": 10.0, "shares_outstanding": 100.0})
    assert e._get_metric_with_year("EBITDA") == (200.0, "2024A")


def test_net_debt_uses_balance_sheet_value_or_falls_back():
    assert engine(net_debt=-50.0)._net_debt(1000.0)[0] == -50.0
    value, label = engine()._net_debt(1000.0)
    assert value == pytest.approx(100.0)
    assert label == "assumed 10% of EV"
    assert engine(net_debt=float("nan"))._net_debt(1000.0)[0] == pytest.approx(100.0)


def test_ev_ebitda_uses_latest_actual_ebitda_historical_multiple_and_net_debt():
    km = pd.DataFrame({"enterpriseValueOverEBITDA": [10.0, 12.0, -3.0, np.nan]})  # negatives ignored
    r = engine(key_metrics=km, net_debt=100.0).calculate_ev_ebitda_valuation()
    # EV = 230 x 11 = 2530; equity = 2530 - 100 = 2430; per share = 24.30
    assert r.target_price == pytest.approx(24.30)
    assert r.assumptions["EBITDA (2025A)"].endswith("230")
    assert r.assumptions["Multiple source"] == "historical average of 2 years"
    assert r.assumptions["Target Multiple"] == "11.0x"


def test_ev_ebitda_without_history_keeps_12x_default():
    r = engine(net_debt=0.0).calculate_ev_ebitda_valuation()
    assert r.target_price == pytest.approx(230 * 12 / 100)
    assert "default 12x" in r.assumptions["Multiple source"]


def test_dcf_prefers_reported_free_cash_flow():
    with_fcf = engine(free_cash_flow=150.0, net_debt=0.0).calculate_dcf_valuation()
    assert with_fcf.assumptions["Base FCF"].startswith("reported FCF")
    without = engine(net_debt=0.0).calculate_dcf_valuation()
    assert without.assumptions["Base FCF"] == "60% of EBITDA (2025A)"
    # Reported FCF 150 > 0.6 x 230 = 138, so its value is higher
    assert with_fcf.target_price > without.target_price


def test_load_valuation_inputs_reads_latest_rows(tmp_path):
    from create_equity_report import load_valuation_inputs
    pd.DataFrame({"date": ["2025-03-31", "2026-03-31"], "netDebt": [10.0, -20.0]}).to_csv(
        tmp_path / "balance_sheet_raw_data.csv", index=False)
    pd.DataFrame({"date": ["2026-03-31", "2025-03-31"], "freeCashFlow": [55.0, 40.0]}).to_csv(
        tmp_path / "cash_flow_raw_data.csv", index=False)
    pd.DataFrame({"year": [2025, 2026], "enterpriseValueOverEBITDA": [15.0, 17.0]}).to_csv(
        tmp_path / "key_metrics_raw_data.csv", index=False)
    inputs = load_valuation_inputs(str(tmp_path))
    assert inputs["net_debt"] == -20.0
    assert inputs["free_cash_flow"] == 55.0
    assert list(inputs["key_metrics"]["enterpriseValueOverEBITDA"]) == [15.0, 17.0]
    assert load_valuation_inputs(str(tmp_path / "missing")) == {}
