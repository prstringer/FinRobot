"""Tests for modules.technical_analysis on synthetic price series."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules import technical_analysis as ta  # noqa: E402
from modules.technical_charts import make_technical_chart  # noqa: E402


def _frame(close, volume=None):
    close = np.asarray(close, dtype=float)
    n = len(close)
    return pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=n), "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": np.full(n, 1000.0) if volume is None else np.asarray(volume, dtype=float)})


def _path(points):
    """Piecewise-linear series through (index, price) key points; key points are unique extremes (no tied joins)."""
    xs, ys = zip(*points)
    return np.interp(np.arange(xs[-1] + 1), xs, ys)


def _diverging_uptrend():
    """Long rise, a sharp first peak, a pull-back, then a slow, weaker climb to a higher high."""
    return _frame(_path([(0, 100), (150, 130), (170, 190), (195, 165), (255, 196), (262, 193)]))


def test_short_history_is_flagged_unavailable():
    f = ta.compute_technical_facts(_frame(np.linspace(100, 110, 20)))
    assert f["available"] is False and "fewer than" in f["reason"]
    assert ta.readings_table_rows(f)[0][1] == "n/a" and ta.tests_rows(f) == []


def test_rsi_zones_follow_c4_boundaries():
    assert ta.rsi_zone(25) == "oversold" and ta.rsi_zone(45) == "neutral" and ta.rsi_zone(59.9) == "neutral"
    assert ta.rsi_zone(60) == "strong, approaching overbought" and ta.rsi_zone(68) == "strong, approaching overbought"
    assert ta.rsi_zone(70.5) == "overbought" and ta.rsi_zone(None) == "n/a"
    assert "bullish" not in ta.rsi_zone(65)


def test_readings_are_named_and_consistent():
    df = _diverging_uptrend()
    f = ta.compute_technical_facts(df)
    c = df["close"]
    assert f["price"] == round(c.iloc[-1], 2)
    assert f["sma20"] == pytest.approx(c.tail(20).mean(), abs=0.011) and f["sma50"] == pytest.approx(c.tail(50).mean(), abs=0.011)
    assert f["sma200"] == pytest.approx(c.tail(200).mean(), abs=0.011)
    assert f["price_vs_sma200_pct"] == round((c.iloc[-1] / c.tail(200).mean() - 1) * 100, 1)
    assert f["week52_high"] == round(df["high"].tail(250).max(), 2)
    assert 0 <= f["week52_position_pct"] <= 100
    assert f["rsi14"] is not None and 0 <= f["rsi14"] <= 100
    assert f["macd_histogram"] == pytest.approx(f["macd"] - f["macd_signal"], abs=0.02)
    assert f["rsi_days_above_70_90d"] >= 1
    assert "sessions" in f["volume_trend_window"] and "median" in f["volume_trend_window"]


def test_no_200_day_average_with_under_200_sessions():
    f = ta.compute_technical_facts(_frame(np.linspace(100, 150, 120)))
    assert f["sma200"] is None and f["price_vs_sma200_pct"] is None and f["crossover_50_200"] is None
    trend = next(t for t in f["tests"] if t["test"] == "Trend")
    assert trend["state"] == "Up" and "fewer than 200" in trend["evidence"]


def test_bearish_divergence_found_on_price_higher_high_with_weaker_macd():
    df = _diverging_uptrend()
    f = ta.compute_technical_facts(df)
    bear = f["divergence"]["bearish"]
    assert bear and {d["indicator"] for d in bear} >= {"MACD"}
    d = next(x for x in bear if x["indicator"] == "MACD")
    assert d["price_2"] > d["price_1"] and d["indicator_2"] < d["indicator_1"]
    assert d["price_date_2"] > d["price_date_1"]
    assert f["divergence"]["bullish"] == []
    assert f["divergence_sentence"].startswith("Bearish divergence")
    assert "momentum fading against price" in f["verdict_sentence"]


def test_no_divergence_in_a_steady_trend():
    f = ta.compute_technical_facts(_frame(np.linspace(100, 220, 300)))
    assert f["divergence"]["bearish"] == [] and f["divergence"]["bullish"] == []
    assert f["divergence_sentence"].startswith("No divergence")


def test_bullish_divergence_on_lower_low_with_higher_indicator_low():
    f = ta.compute_technical_facts(_frame(_path([(0, 200), (90, 120), (110, 150), (180, 118), (200, 126)])))
    bull = f["divergence"]["bullish"]
    assert bull and all(d["price_2"] < d["price_1"] and d["indicator_2"] > d["indicator_1"] for d in bull)


def test_downtrend_tests_and_death_cross():
    f = ta.compute_technical_facts(_frame(np.concatenate([np.linspace(100, 160, 160), np.linspace(160, 80, 160)])))
    states = {t["test"]: t["state"] for t in f["tests"]}
    assert states["Trend"] == "Down"
    assert f["crossover_50_200"]["direction"] == "death"
    assert f["verdict_sentence"].startswith("Downtrend")


def test_four_independent_tests_present_and_no_vote_count():
    f = ta.compute_technical_facts(_diverging_uptrend())
    assert [t["test"] for t in f["tests"]] == ["Trend", "Momentum", "Participation", "Stretch"]
    assert all(t["state"] and t["evidence"] for t in f["tests"])
    assert "vote" not in f["verdict_sentence"].lower()


def test_participation_weak_when_volume_dries_up_in_the_rally():
    n = 320
    close = np.concatenate([np.full(200, 100.0), np.linspace(101, 200, n - 200)])
    vol = np.concatenate([np.full(200, 5000.0), np.full(n - 200, 1000.0)])
    f = ta.compute_technical_facts(_frame(close, vol))
    part = next(t for t in f["tests"] if t["test"] == "Participation")
    assert part["state"] == "Weak" and f["volume_trend_vs_base"] < 0.85
    assert "thinly traded" in f["verdict_sentence"]


def test_stretch_levels_and_verdict_for_late_stretched_uptrend():
    close = np.concatenate([np.full(200, 100.0), np.linspace(101, 190, 100)])
    f = ta.compute_technical_facts(_frame(close))
    assert f["price_vs_sma200_pct"] > 40
    assert next(t for t in f["tests"] if t["test"] == "Stretch")["state"] == "Very high"
    assert f["verdict_sentence"].startswith("Uptrend intact, but late and stretched")


def test_median_volume_ignores_a_single_block_trade_in_the_base():
    n = 300
    close = np.concatenate([np.full(150, 100.0), np.linspace(100, 160, n - 150)])
    vol = np.full(n, 1000.0)
    vol[100] = 5_000_000          # one-off block trade in the base
    f = ta.compute_technical_facts(_frame(close, vol))
    assert f["volume_trend_vs_base"] == pytest.approx(1.0, abs=0.05)


def test_text_has_no_em_dashes_and_rows_are_complete():
    f = ta.compute_technical_facts(_diverging_uptrend(), eps_info=dict(eps_fy_now=2.0, eps_fy_prev=6.0, eps_ttm=4.0))
    rows = ta.readings_table_rows(f)
    assert len(rows) == 12 and all(len(r) == 3 for r in rows)
    blob = " ".join(" ".join(map(str, r)) for r in rows + ta.tests_rows(f)) + f["verdict_sentence"] + f["divergence_sentence"] + f["price_vs_earnings"]["sentence"]
    assert "\u2014" not in blob and "\u2013" not in blob


def test_price_vs_earnings_ahead_behind_in_line_and_negative_base():
    df = _frame(np.linspace(100, 200, 300))                       # +~60% over a year
    ahead = ta.price_vs_earnings(df, eps_fy_now=2.0, eps_fy_prev=2.0, eps_ttm=2.0)
    assert ahead["price_vs_earnings"] == "ahead" and ahead["eps_ttm_change_pct"] == 0
    behind = ta.price_vs_earnings(df, eps_fy_now=2.0, eps_fy_prev=6.0, eps_ttm=5.0, eps_ttm_year_ago=2.0)
    assert behind["price_vs_earnings"] == "behind" and behind["eps_ttm_change_pct"] == 150.0
    assert behind["eps_fy_change_pct"] == pytest.approx(-66.7, abs=0.1)
    inline = ta.price_vs_earnings(df, eps_fy_now=3.0, eps_fy_prev=2.0, eps_ttm=3.4, eps_ttm_year_ago=2.0)
    assert inline["price_vs_earnings"] == "in line"
    neg = ta.price_vs_earnings(df, eps_fy_now=1.0, eps_fy_prev=-1.0, eps_ttm=None)
    assert neg["eps_fy_change_pct"] is None and neg["price_vs_earnings"] == "n/a"
    short = ta.price_vs_earnings(_frame(np.linspace(100, 110, 100)), 2.0, 2.0, 2.0)
    assert short["price_change_1y_pct"] is None and "Less than a year" in short["sentence"]


def test_facts_with_eps_info_embed_strip():
    f = ta.compute_technical_facts(_diverging_uptrend(), eps_info=dict(eps_fy_now=2.0, eps_fy_prev=6.0, eps_ttm=4.0))
    assert "price_vs_earnings" in f and f["price_vs_earnings"]["sentence"]


def test_chart_is_written_and_marks_divergence(tmp_path):
    df = _diverging_uptrend()
    f = ta.compute_technical_facts(df)
    out = make_technical_chart(df, f, str(tmp_path / "t.png"), "TEST")
    assert out and os.path.getsize(out) > 10_000
    assert make_technical_chart(df, {"available": False}, str(tmp_path / "x.png")) is None
