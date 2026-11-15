"""Tests for cash-flow facts/charts (Part D) and per-ratio bars (Part E)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules import cashflow_analysis as ca  # noqa: E402
from modules import cashflow_charts as cc  # noqa: E402
from modules import ratio_bars as rb  # noqa: E402

CR = 1e7
WELSPUN_DIR = ("/private/tmp/claude-501/-Users-ramjoshi-Coding-Projects-FinRobot/"
               "5b26141b-d31e-44a5-aefc-feaa847a9947/scratchpad/run2/analysis")


def _synthetic(conv=(0.9, 0.9, 0.9, 0.9, 0.9), rev=(100, 110, 120, 130, 140), capex=(10, 10, 10, 10, 10), buyback=None):
    years = [2022, 2023, 2024, 2025, 2026]
    inc = pd.DataFrame([{"year": y, "revenue": r * CR, "ebitda": 20 * CR, "costOfRevenue": 50 * CR,
                         "sellingGeneralAndAdministrativeExpenses": 20 * CR} for y, r in zip(years, rev)])
    bal = pd.DataFrame([{"year": y, "inventory": 30 * CR, "netReceivables": 15 * CR, "accountPayables": 12 * CR} for y in years])
    cf = pd.DataFrame([{"year": y, "date": pd.Timestamp(f"{y}-03-31"), "operatingCashFlow": 20 * c * CR,
                        "capitalExpenditure": -k * CR, "netCashUsedForInvestingActivites": -(k + 2) * CR,
                        "dividendsPaid": -3 * CR, "netChangeInCash": 1 * CR}
                       for y, c, k in zip(years, conv, capex)])
    if buyback is not None:
        cf["buybacks"] = -buyback * CR
        cf["debtProceeds"] = 5 * CR
        cf["debtRepayment"] = -8 * CR
    return inc, bal, cf


def test_basic_flows_conversion_and_signs():
    f = ca.compute_cashflow_facts(*_synthetic())
    assert f["years"] == ["2022A", "2023A", "2024A", "2025A", "2026A"]       # oldest to newest
    r = f["per_year"]["2026A"]
    assert r["cfo"] == pytest.approx(18.0) and r["capex"] == pytest.approx(-10.0)
    assert r["fcf"] == pytest.approx(8.0)                                      # CFO less capex
    assert r["other_investing"] == pytest.approx(-2.0)                         # investing total minus capex
    assert r["dividends"] == pytest.approx(-3.0)
    assert r["conversion"] == pytest.approx(0.9)
    assert f["aggregate"]["conversion"] == pytest.approx(0.9)


def test_investing_cash_flow_is_never_shown_as_capex():
    inc, bal, cf = _synthetic()
    cf["capitalExpenditure"] = 0.0                                             # normaliser's "missing" marker
    f = ca.compute_cashflow_facts(inc, bal, cf)
    assert all(f["per_year"][y]["capex"] is None and f["per_year"][y]["fcf"] is None for y in f["years"])
    assert not f["capex_available"] and "NOT shown as capex" in f["capex_basis"]
    assert any("capex" in u for u in f["unavailable"])


def test_optional_columns_flagged_unavailable_then_used():
    f = ca.compute_cashflow_facts(*_synthetic())
    assert any("buybacks" in u for u in f["unavailable"]) and any("net debt" in u for u in f["unavailable"])
    g = ca.compute_cashflow_facts(*_synthetic(buyback=4))
    assert g["per_year"]["2026A"]["buybacks"] == pytest.approx(-4.0)
    assert g["per_year"]["2026A"]["debt_net"] == pytest.approx(-3.0)
    assert g["uses_of_cash"]["buybacks"] == pytest.approx(-20.0)
    assert g["uses_of_cash"]["debt_repaid_gross"] == pytest.approx(-40.0)
    assert g["uses_of_cash"]["cash_retained"] == pytest.approx(5.0)
    assert not any("buybacks" in u for u in g["unavailable"])


def test_working_capital_days():
    f = ca.compute_cashflow_facts(*_synthetic())
    w = f["working_capital_days"]["2026A"]
    assert w["inventory_days"] == pytest.approx(30 / 50 * 365)
    assert w["receivable_days"] == pytest.approx(15 / 140 * 365)
    assert w["payable_days"] == pytest.approx(12 / 50 * 365)
    assert w["cash_conversion_days"] == pytest.approx(w["inventory_days"] + w["receivable_days"] - w["payable_days"])


def test_countercyclical_flag_and_verdicts():
    # revenue falls in 2023 and 2026 while conversion jumps
    f = ca.compute_cashflow_facts(*_synthetic(conv=(0.4, 1.0, 0.4, 0.5, 1.5), rev=(100, 85, 100, 110, 98)))
    cyc = f["cyclicality"]
    assert cyc["flag"] and cyc["correlation"] < -0.4
    assert cyc["avg_conversion_when_revenue_fell"] > cyc["avg_conversion_when_revenue_grew"]
    assert "partly one-off" in cyc["statement"]
    assert f["answers"]["flattered_by_one_offs"]["verdict"] == "concern"
    flat = ca.compute_cashflow_facts(*_synthetic())
    assert flat["answers"]["flattered_by_one_offs"]["verdict"] == "good"
    assert flat["answers"]["profit_to_cash"]["verdict"] == "good"


def test_funds_itself_rules():
    strong = ca.compute_cashflow_facts(*_synthetic(conv=(1.2,) * 5, capex=(5,) * 5))
    assert strong["answers"]["funds_itself"]["verdict"] == "good"
    weak = ca.compute_cashflow_facts(*_synthetic(conv=(0.5,) * 5, capex=(12,) * 5))
    assert weak["answers"]["funds_itself"]["verdict"] == "concern"


def test_verdict_shape_word_limit_and_no_em_dash():
    f = ca.compute_cashflow_facts(*_synthetic(conv=(0.4, 1.0, 0.4, 0.5, 1.5), rev=(100, 85, 100, 110, 98)))
    for a in f["answers"].values():
        assert set(a) == {"verdict", "yardstick", "so_what", "source_note"}
        assert a["verdict"] in ("good", "neutral", "concern")
        assert a["yardstick"] in ("own_history", "peers", "general_norm")
        assert len(a["so_what"].split()) <= 25
        for dash in ("—", "–"):
            assert dash not in a["so_what"] and dash not in a["source_note"]
        if a["yardstick"] == "general_norm":
            assert "general norm, not company-specific" in a["source_note"]
    for line in ca.cashflow_text_lines(f):
        assert "—" not in line


def test_table_rows_chronological_with_na_and_ttm():
    inc, bal, cf = _synthetic()
    ttm = {"ebitda": 22 * CR}
    ttm_cf = {"operatingCashFlow": 21 * CR, "capitalExpenditure": -9 * CR, "netCashUsedForInvestingActivites": -11 * CR,
              "dividendsPaid": -3 * CR, "netChangeInCash": 2 * CR}
    f = ca.compute_cashflow_facts(inc, bal, cf, ttm=ttm, ttm_cashflow=ttm_cf)
    header, rows = ca.cashflow_table_rows(f)
    assert header == ["Rs crore", "2022A", "2023A", "2024A", "2025A", "2026A", "TTM"]
    labels = [r[0] for r in rows]
    assert "Capex (fixed and intangible assets)" in labels and "Share buybacks" in labels
    bb = next(r for r in rows if r[0] == "Share buybacks")
    assert set(bb[1:]) == {"n/a"}
    capex = next(r for r in rows if r[0].startswith("Capex"))
    assert capex[1] == "(10)"                                                  # outflows in brackets
    assert f["per_year"]["TTM"]["conversion"] == pytest.approx(21 / 22)
    assert "TTM" in "".join(ca.cashflow_text_lines(f))


def test_derive_ttm_cashflow_from_half_years():
    fy = pd.DataFrame([{"date": pd.Timestamp("2026-03-31"), "operatingCashFlow": 100.0, "capitalExpenditure": -40.0,
                        "netCashUsedForInvestingActivites": -50.0, "dividendsPaid": -5.0, "netChangeInCash": 3.0}])
    half = pd.DataFrame([{"date": pd.Timestamp("2026-09-30"), "operatingCashFlow": 70.0, "capitalExpenditure": -20.0,
                          "netCashUsedForInvestingActivites": -25.0, "dividendsPaid": -2.0, "netChangeInCash": 4.0},
                         {"date": pd.Timestamp("2025-09-30"), "operatingCashFlow": 30.0, "capitalExpenditure": -25.0,
                          "netCashUsedForInvestingActivites": -26.0, "dividendsPaid": -1.0, "netChangeInCash": 1.0}])
    t = ca.derive_ttm_cashflow(fy, half)
    assert t["operatingCashFlow"] == pytest.approx(140.0) and t["capitalExpenditure"] == pytest.approx(-35.0)
    assert ca.derive_ttm_cashflow(fy, half.iloc[[1]]) is None                   # halves do not line up


def test_missing_cash_flow_degrades_gracefully():
    f = ca.compute_cashflow_facts(_synthetic()[0], None, None)
    assert f["available"] is False and ca.cashflow_table_rows(f) == (["Rs crore"], [])
    assert "not available" in ca.cashflow_text_lines(f)[0]


def test_charts_render_chronological(tmp_path):
    f = ca.compute_cashflow_facts(*_synthetic(buyback=4))
    out = cc.draw_cashflow_charts(f, str(tmp_path), "TEST")
    for k, p in out.items():
        assert p and os.path.getsize(p) > 5000, k
    assert cc.cash_waterfall_chart({"available": False}, str(tmp_path / "x.png")) is None


# ---------------------------------------------------------------------------- ratio bars
def _ratios(vals):
    """vals: dict column -> list of 5 values (oldest to newest)."""
    n = len(next(iter(vals.values())))
    d = {"calendarYear": list(range(2022, 2022 + n)), "year": list(range(2022, 2022 + n))}
    d.update(vals)
    return pd.DataFrame(d).iloc[::-1].reset_index(drop=True)                   # newest first like the CSV


def test_ratio_directions_and_own_history_verdicts():
    df = _ratios({"grossProfitMargin": [0.40, 0.41, 0.42, 0.43, 0.46], "debtEquityRatio": [0.9, 0.8, 0.7, 0.6, 0.4],
                  "returnOnEquity": [0.15, 0.14, 0.13, 0.12, 0.05]})
    f = rb.compute_ratio_bar_facts(df)
    assert f["gross_margin"]["verdict"] == "good" and f["gross_margin"]["yardstick"] == "own_history"   # top of own range
    assert f["debt_equity"]["verdict"] == "good" and f["debt_equity"]["direction"] == "lower is better"  # lowest = best
    assert f["roe"]["verdict"] == "concern"                                                              # bottom of own range
    assert set(f["roe"]["scale"]) == {"value", "range_min", "range_max", "peer_median"}
    assert f["roe"]["scale"]["value"] == pytest.approx(5.0)                  # shown as percent
    assert f["roe"]["scale"]["range_max"] == pytest.approx(15.0)


def test_peers_take_precedence_and_direction_is_respected():
    df = _ratios({"returnOnEquity": [0.10] * 5, "debtEquityRatio": [1.0] * 5})
    f = rb.compute_ratio_bar_facts(df, peer_medians={"roe": 20.0, "debt_equity": 2.0})
    assert f["roe"]["yardstick"] == "peers" and f["roe"]["verdict"] == "concern"      # below peers, higher is better
    assert f["debt_equity"]["verdict"] == "good"                                       # below peers, lower is better
    near = rb.compute_ratio_bar_facts(df, peer_medians={"roe": 10.5})
    assert near["roe"]["verdict"] == "neutral"                                         # within 10% of the median


def test_general_norm_only_when_no_history_and_is_labelled():
    df = _ratios({"currentRatio": [0.8], "interestCoverage": [5.0], "returnOnEquity": [0.1]})
    f = rb.compute_ratio_bar_facts(df)
    assert f["current_ratio"]["yardstick"] == "general_norm" and f["current_ratio"]["verdict"] == "concern"
    assert f["interest_coverage"]["verdict"] == "good"
    assert "general norm, not company-specific" in f["current_ratio"]["source_note"]
    assert f["roe"]["verdict"] is None and f["roe"]["yardstick"] is None              # no yardstick, no verdict


def test_only_available_ratios_and_sga_from_income():
    df = _ratios({"grossProfitMargin": [0.4, 0.4, 0.4, 0.4, 0.4]})
    inc = pd.DataFrame({"year": [2022, 2023, 2024], "revenue": [100.0, 100.0, 100.0],
                        "sellingGeneralAndAdministrativeExpenses": [30.0, 33.0, 38.0]})
    f = rb.compute_ratio_bar_facts(df, income_df=inc)
    assert set(f) == {"gross_margin", "sga_pct"}
    assert f["sga_pct"]["verdict"] == "concern"                                        # highest SG&A share is worst
    for r in rb.ratio_bar_rows(f):
        assert len(r["so_what"].split()) <= 25 and "—" not in r["so_what"]


def test_peer_medians_exclude_subject():
    mk = lambda v: _ratios({"returnOnEquity": [v] * 3})
    m = rb.peer_medians_from_ratios({"AAA": mk(0.30), "BBB": mk(0.10), "CCC": mk(0.20), "SUBJ": mk(0.90)}, "subj")
    assert m["roe"] == pytest.approx(20.0)


def test_draw_ratio_bars(tmp_path):
    df = _ratios({"grossProfitMargin": [0.4, 0.41, 0.42, 0.43, 0.46], "returnOnEquity": [0.15, 0.14, 0.13, 0.12, 0.05],
                  "debtEquityRatio": [0.9, 0.8, 0.7, 0.6, 0.4]})
    f = rb.compute_ratio_bar_facts(df, peer_medians={"gross_margin": 40.0})
    p = rb.draw_ratio_bars(f, str(tmp_path / "bars.png"))
    assert p and os.path.getsize(p) > 5000
    assert rb.draw_ratio_bars({}, str(tmp_path / "none.png")) is None
    lines = rb.ratio_bar_text_lines(f)
    assert any("peer median 40.0%" in l for l in lines) and any("lower is better" in l for l in lines)


# ---------------------------------------------------------------------------- WELSPUNLIV
@pytest.mark.skipif(not os.path.exists(os.path.join(WELSPUN_DIR, "cash_flow_raw_data.csv")), reason="scratch WELSPUNLIV data absent")
def test_welspunliv_fixture():
    rd = lambda n: pd.read_csv(os.path.join(WELSPUN_DIR, n))
    inc, bal, cf, ratios = rd("income_statement_raw_data.csv"), rd("balance_sheet_raw_data.csv"), rd("cash_flow_raw_data.csv"), rd("ratios_raw_data.csv")
    f = ca.compute_cashflow_facts(inc, bal, cf)
    conv = {y: f["per_year"][y]["conversion"] for y in f["years"]}
    assert conv["2023A"] == pytest.approx(1.00, abs=0.01) and conv["2026A"] == pytest.approx(1.48, abs=0.01)
    for y in ("2022A", "2024A", "2025A"):
        assert 0.35 < conv[y] < 0.55
    assert f["cyclicality"]["flag"] and f["cyclicality"]["years_revenue_fell"] == ["2023A", "2026A"]
    assert f["per_year"]["2026A"]["capex"] == pytest.approx(-798.6, rel=1e-3)
    assert f["per_year"]["2026A"]["fcf"] == pytest.approx(376.4, rel=1e-3)
    assert f["answers"]["flattered_by_one_offs"]["verdict"] == "concern"
    assert f["answers"]["profit_to_cash"]["verdict"] == "neutral"
    header, rows = ca.cashflow_table_rows(f)
    assert header[1:] == ["2022A", "2023A", "2024A", "2025A", "2026A"]

    bars = rb.compute_ratio_bar_facts(ratios, income_df=inc)
    assert {"gross_margin", "ebitda_margin", "net_margin", "roe", "roa", "current_ratio", "interest_coverage",
            "debt_equity", "sga_pct"} <= set(bars)
    assert bars["sga_pct"]["verdict"] == "concern"                           # 37.8% is the worst of five years
    assert bars["roe"]["verdict"] == "concern" and bars["debt_equity"]["verdict"] == "good"
