"""Fiscal calendar, reporting-basis guard and sentiment gate."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules import fiscal_guard as fg  # noqa: E402
from modules import sentiment_gate as sg  # noqa: E402


# ---------------------------------------------------------------- fiscal calendar
def test_indian_fiscal_year_labels_and_span():
    assert fg.fy_label("2026A") == "FY26" and fg.fy_label(2026) == "FY26" and fg.fy_label("2027E") == "FY27E"
    assert fg.fy_span(2026) == "1 Apr 2025 to 31 Mar 2026"


def _frame(dates_years):
    return pd.DataFrame([{"date": pd.Timestamp(d), "year": y} for d, y in dates_years])


def test_march_year_ends_pass():
    f = _frame([("2026-03-31", 2026), ("2025-03-31", 2025)])
    assert fg.check_fiscal_year_ends({"income": f, "balance": f}) == []


def test_a_period_ending_in_november_is_flagged_not_labelled_as_a_fiscal_year():
    f = _frame([("2026-11-30", 2026), ("2025-03-31", 2025)])
    issues = fg.check_fiscal_year_ends({"income": f})
    assert len(issues) == 1 and "2026-11-30" in issues[0] and "not 31 March" in issues[0]


def test_year_number_must_match_the_period_end_year():
    issues = fg.check_fiscal_year_ends({"income": _frame([("2026-03-31", 2025)])})
    assert issues and "does not match" in issues[0]


def test_ttm_is_never_a_full_year_column_in_the_real_mahamaya_table():
    """The real MAHASTEEL table: A columns are FY-end years only; the TTM period (Sep 2026) is not among them."""
    path = os.path.join(os.path.dirname(__file__), "..", "output", "MAHASTEEL", "analysis", "financial_metrics_and_forecasts.csv")
    if not os.path.exists(path):
        pytest.skip("MAHASTEEL output not present")
    cols = list(pd.read_csv(path, nrows=0).columns)
    assert "TTM" not in cols and [c for c in cols if c.endswith("A")] == ["2022A", "2023A", "2024A", "2025A", "2026A"]


# ---------------------------------------------------------------- reporting basis
class _Client:
    """Stub FinEdge client: quarterly P&L rows for the consolidated and standalone bases."""

    def __init__(self, c_now, c_prior, s_now, s_prior):
        self.q = {"c": (c_now, c_prior), "s": (s_now, s_prior)}

    def financials(self, symbol, code, period, st):
        now, prior = self.q[st]

        def row(end, pat):
            return {"period_end": end, "revenueFromOperations": 2.5e9, "expenses": 2.4e9, "financeCosts": 1e7,
                    "depreciationAndAmortisation": 2e7, "profitOrLossAttributableToOwners": pat, "year": int(str(end)[:4])}
        return [row(20260930, now), row(20250930, prior)]


def test_quarter_basis_comparison_shows_both_bases_and_flags_divergence():
    b = fg.quarter_basis_comparison(_Client(2.59e7, 1.88e7, 2.56e7, 1.40e7), "X", "c")
    assert b["used"] == "consolidated"
    assert b["consolidated"]["yoy_pct"] == pytest.approx(37.8, abs=0.2)
    assert b["standalone"]["yoy_pct"] == pytest.approx(82.9, abs=0.2)
    assert b["diverges"] is True
    block = fg.basis_fact_block(b, 2026)
    assert "CONSOLIDATED basis" in block and "FY26 is 1 Apr 2025 to 31 Mar 2026" in block
    assert "+83% YoY" in block and "+38% YoY" in block and "never mix" in block
    warn = fg.basis_warnings(b, [])
    assert warn and "+83% standalone" in warn[0]


def test_no_divergence_when_bases_agree():
    b = fg.quarter_basis_comparison(_Client(2.6e7, 2.0e7, 2.6e7, 2.0e7), "X", "c")
    assert b["diverges"] is False and fg.basis_warnings(b, []) == []


# ---------------------------------------------------------------- sentiment gate
def _ctx(price=1519.9, grid=(268, 316, 363, 411)):
    return {"market": {"price": price}, "fair_value": {"grid": [{"multiple": 12 + 2 * i, "implied_price": p} for i, p in enumerate(grid)]}}


def _news(overall="positive"):
    return {"sentiment_summary": {"overall": overall, "positive": 4, "negative": 0, "neutral": 4}}


def test_gate_triggers_on_mahamaya_numbers_and_overrides_a_positive_tone():
    news = _news("positive")
    st = sg.apply_gate(news, _ctx())
    ss = news["sentiment_summary"]
    assert st["triggered"] and st["implied_price"] == pytest.approx(339.5)
    assert ss["overall"] == "contradictory/bearish" and ss["news_tone"] == "positive"
    assert "below the current price" in ss["gate_note"] and "70%" in ss["gate_note"]
    assert sg.display_label(ss["overall"]) == "CONTRADICTORY/BEARISH"


def test_gate_boundary_exactly_seventy_percent_does_not_trigger():
    price = 1000.0
    assert sg.gate_status(_ctx(price, grid=(700.0,)))["triggered"] is False        # implied == 0.70 x price
    assert sg.gate_status(_ctx(price, grid=(699.9,)))["triggered"] is True


def test_gate_leaves_sentiment_alone_when_valuation_is_supportive():
    news = _news("positive")
    st = sg.apply_gate(news, _ctx(price=300.0))
    assert st["triggered"] is False and news["sentiment_summary"]["overall"] == "positive"
    assert "gate_note" not in news["sentiment_summary"]


def test_gate_clears_a_previous_override_when_valuation_recovers():
    news = _news("positive")
    sg.apply_gate(news, _ctx())
    sg.apply_gate(news, _ctx(price=300.0))
    assert news["sentiment_summary"]["overall"] == "positive"


def test_gate_is_unavailable_without_a_fair_value_and_does_nothing():
    news = _news("positive")
    st = sg.apply_gate(news, {"market": {"price": 100}, "fair_value": {"grid": []}})
    assert st["available"] is False and news["sentiment_summary"]["overall"] == "positive"
    assert sg.apply_gate(None, _ctx())["triggered"] is True                          # no news object: status only


def test_gate_fact_block_reaches_the_text_layer_only_when_triggered():
    st = sg.gate_status(_ctx())
    assert "SENTIMENT GATE" in sg.gate_fact_block(st, "positive") and "conflict" in sg.gate_fact_block(st, "positive")
    assert sg.gate_fact_block(sg.gate_status(_ctx(price=300.0)), "positive") is None
