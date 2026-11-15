"""Tests for the visual-report chart specs, built from the committed WELSPUNLIV fixture."""
import copy
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from report_visual import charts  # noqa: E402
from report_visual import charts_common as cc  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "visual_welspun")


@pytest.fixture(scope="module")
def ctx():
    with open(os.path.join(FIX, "report_context.json"), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def panels(ctx):
    return charts.build_panel_charts(ctx, FIX)


def _all(panels):
    return [(pid, c) for pid, cs in panels.items() for c in cs]


def _chart(panels, cid):
    return next(c for _, c in _all(panels) if c["id"] == cid)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


# ---------------------------------------------------------------- structure
def test_every_panel_has_the_expected_charts_and_roles(panels):
    got = {pid: [(c["id"], c["role"]) for c in cs] for pid, cs in panels.items()}
    assert got["profit_engine"] == [("margin_scissors", "main"), ("margin_waterfall", "small")]
    assert got["cyclicality"] == [("cyclicality", "main")]
    assert got["strength"] == [("cash_conversion", "main"), ("debt_and_coverage", "small")]
    assert got["valuation"] == [("valuation_strip", "main"), ("pe_vs_history", "small"), ("price_ruler", "small")]
    assert got["price"] == [("price_trend", "main"), ("momentum", "small")]
    assert got["summary"] == [("price_vs_earnings", "small")]
    assert got["watch"] == []
    assert list(panels) == charts.PANEL_IDS
    assert charts.LAST_ERRORS == []


def test_one_main_chart_per_panel_that_has_charts_and_at_most_two_small(panels):
    for pid, cs in panels.items():
        mains = [c for c in cs if c["role"] == "main"]
        assert len(mains) <= 1
        assert len([c for c in cs if c["role"] == "small"]) <= 2 + (1 if pid == "valuation" else 0)
        if pid not in ("summary", "watch"):
            assert len(mains) == 1


def test_chart_envelope_matches_the_schema(panels):
    for _, c in _all(panels):
        assert {"id", "role", "title", "subtitle", "option", "alt", "datatable", "variants", "default_variant",
                "static_note", "height"} <= set(c)
        assert c["alt"].strip() and c["height"] > 0
        assert c["datatable"]["columns"] and c["datatable"]["rows"]
        json.dumps(c, allow_nan=False)


def test_every_option_and_variant_validates_and_uses_only_tokens(panels):
    for _, c in _all(panels):
        cc.validate_option(c["option"])
        for v in c["variants"] or []:
            cc.validate_option(v["option"])
        assert c["option"]["animation"] is False


def test_no_radar_pie_or_verdict_colours_in_any_chart(panels):
    text = json.dumps([c for _, c in _all(panels)])
    for banned in ('"pie"', '"radar"', '"gauge"', "bar3D", '"@good"', '"@neutral"', '"@concern"'):
        assert banned not in text, banned
    for node in _walk([c["option"] for _, c in _all(panels)]):
        assert node.get("type") not in ("pie", "radar", "gauge")


def test_total_size_leaves_room_for_the_html_budget(panels):
    size = len(json.dumps([c for _, c in _all(panels)]))
    assert size < 600_000      # whole chart payload; ECharts itself is ~1 MB of the 2 MB HTML budget


# ---------------------------------------------------------------- validator
def _opt(**kw):
    o = {"animation": False, "series": []}
    o.update(kw)
    return o


@pytest.mark.parametrize("bad", [
    {"animation": True},
    {"textStyle": {"color": "#ff0000"}},
    {"series": [{"type": "line", "itemStyle": {"color": "red"}}]},
    {"series": [{"type": "pie"}]},
    {"series": [{"type": "line", "label": {"formatter": "function(p){return p}"}}]},
    {"series": [{"type": "line", "data": [float("nan")]}]},
    {"series": [{"type": "line", "itemStyle": {"color": "@good"}}]},
    {"radar": {"indicator": []}},
    {"series": [{"type": "line", "lineStyle": {"color": "rgba(0,0,0,0.5)"}}]},
])
def test_validate_option_rejects(bad):
    with pytest.raises(ValueError):
        cc.validate_option(_opt(**bad))


def test_validate_option_accepts_tokens_and_unserialisable_is_rejected():
    cc.validate_option(_opt(series=[{"type": "bar", "itemStyle": {"color": "@accent"}, "areaStyle": {"color": "@grey2", "opacity": 0.3}}]))
    with pytest.raises(ValueError):
        cc.validate_option(_opt(series=[{"type": "line", "formatter": (lambda x: x)}]))


# ---------------------------------------------------------------- formats
def test_indian_formats_and_fiscal_labels():
    assert cc.indian(1234567) == "12,34,567"
    assert cc.indian(9399.1) == "9,399"
    assert cc.indian(-123456.78, 2) == "-1,23,456.78"
    assert cc.fmt_cr(9399.1) == "Rs 9,399 cr"
    assert cc.fmt_cr(77.77) == "Rs 77.8 cr"
    assert cc.fmt_cr(250000) == "Rs 2,50,000 cr"
    assert [cc.fy(x) for x in ("2026A", "2027E", "TTM", "2026")] == ["FY26", "FY27E", "TTM", "FY26"]
    assert cc.fmt_x(29.655) == "29.7x" and cc.fmt_pct(37.84) == "37.8%"


def test_split_actual_forecast_joins_at_the_last_actual_point():
    a, f = cc.split_actual_forecast([1, 2, 3, 4], [False, False, True, True])
    assert a == [1, 2, None, None] and f == [None, 2, 3, 4]


# ---------------------------------------------------------------- panel 2
def test_scissors_chronological_forecast_dashed_and_matches_source(panels):
    c = _chart(panels, "margin_scissors")
    opt = c["option"]
    assert opt["xAxis"]["data"] == ["FY22", "FY23", "FY24", "FY25", "FY26", "FY27E", "FY28E", "FY29E"]
    dashed = [s for s in opt["series"] if s.get("lineStyle", {}).get("type") == "dashed"]
    assert len(dashed) == 2 and all("forecast" in s["name"] for s in dashed)
    solid_names = [s["name"] for s in opt["series"] if s["name"] in ("Contribution margin", "SG&A (% of revenue)")]
    assert len(solid_names) == 2
    assert any("markArea" in s and s["markArea"]["data"][0][0]["name"] == "Forecast" for s in opt["series"])
    fin = pd.read_csv(os.path.join(FIX, "financial_metrics_and_forecasts.csv")).set_index("metrics")
    rows = {r[0]: r for r in c["datatable"]["rows"]}
    for yr, lab in (("2022A", "FY22"), ("2026A", "FY26")):
        rev = float(fin.loc["Revenue", yr])
        assert float(rows[lab][1]) == pytest.approx(float(fin.loc["Contribution Profit", yr]) / rev * 100, abs=0.05)
        assert float(rows[lab][2]) == pytest.approx(float(fin.loc["SG&A", yr]) / rev * 100, abs=0.05)
    assert "FY27E (forecast)" in rows
    assert rows["FY22"][2] == "30.7" and rows["FY26"][2] == "37.8" and rows["FY26"][1] == "46.2"


def test_scissors_annotates_the_five_year_high_and_alt_has_start_and_end(panels):
    c = _chart(panels, "margin_scissors")
    labels = [d["label"]["formatter"] for s in c["option"]["series"] for d in s["data"]
              if isinstance(d, dict) and "label" in d]
    assert "SG&A 37.8%, five-year high" in labels
    for token in ("30.7%", "37.8%", "45.3%", "46.2%", "FY22", "FY26"):
        assert token in c["alt"], token


def test_waterfall_compares_best_of_prior_three_years_with_latest(panels, ctx):
    c = _chart(panels, "margin_waterfall")
    cols = c["datatable"]["columns"]
    assert "FY24" in cols[1] and "FY26" in cols[2]
    rows = {r[0]: r for r in c["datatable"]["rows"]}
    assert rows["Revenue"][2] == "9,399" and rows["SG&A"][2] == "3,553" and rows["EBITDA"][2] == "793"
    assert rows["EBITDA"][1] == "1,369"
    assert "peak is FY22" in c["subtitle"]          # across all five actual years FY22 (14.6%) is higher than FY24 (14.1%)
    for token in ("FY24", "FY26", "Rs 9,399 cr", "Rs 793 cr"):
        assert token in c["alt"]
    # bars reconcile: revenue - direct costs = contribution; contribution - SG&A = EBITDA
    assert float(rows["Revenue"][2].replace(",", "")) - float(rows["Direct costs"][2].replace(",", "")) == pytest.approx(
        float(rows["Contribution"][2].replace(",", "")), abs=1.5)


# ---------------------------------------------------------------- panel 3
def test_cyclicality_bars_forecasts_hatched_and_steadiest_peer_band(panels, ctx):
    c = _chart(panels, "cyclicality")
    assert charts.steadiest_peer(ctx)["ticker"] == "KPRMILL"
    opt = c["option"]
    bars = next(s for s in opt["series"] if s["name"] == "Revenue growth")["data"]
    assert bars[0] is None                                   # FY22 has no growth figure
    fc_bars = [b for b in bars[5:] if b]
    assert fc_bars and all("decal" in b["itemStyle"] for b in fc_bars)
    assert all("decal" not in b["itemStyle"] for b in bars[:5] if b)
    assert bars[1]["label"]["formatter"] == "-13.1%" and bars[4]["label"]["formatter"] == "-10.9%"
    band = next(s["markArea"] for s in opt["series"] if s["name"] == "EBITDA")["data"][0]
    kpr = [row["KPRMILL"] for row in ctx["peers"]["ebitda_actual_cr"].values()]
    assert band[0]["yAxis"] == pytest.approx(min(kpr), abs=0.06) and band[1]["yAxis"] == pytest.approx(max(kpr), abs=0.06)
    assert 1218 < band[0]["yAxis"] < 1220 and 1273 < band[1]["yAxis"] < 1275
    assert opt["xAxis"][1]["data"][0] == "FY22" and opt["xAxis"][1]["data"][4] == "FY26"
    for token in ("FY23", "FY26", "KPRMILL", "Rs 753 cr", "Rs 1,369 cr"):
        assert token in c["alt"], token


def test_cyclicality_has_no_dual_axis(panels):
    opt = _chart(panels, "cyclicality")["option"]
    grids = {a["gridIndex"] for a in opt["yAxis"]}
    assert grids == {0, 1}                                   # one value axis per grid, never two on the same grid


# ---------------------------------------------------------------- panel 4
def test_cash_conversion_pairs_labels_and_datatable(panels, ctx):
    c = _chart(panels, "cash_conversion")
    labels = c["option"]["xAxis"]["data"]
    assert labels == ["FY22", "FY23", "FY24", "FY25", "FY26", "TTM"]
    conv = [d["label"]["formatter"] for d in next(s for s in c["option"]["series"] if s["name"] == "Cash conversion")["data"]]
    assert conv[:5] == ["43%", "100%", "39%", "53%", "148%"]
    per = ctx["extras"]["cashflow"]["per_year"]
    rows = {r[0]: r for r in c["datatable"]["rows"]}
    assert rows["FY26"][1] == cc.fmt_cr(per["2026A"]["ebitda"]) and rows["FY26"][2] == cc.fmt_cr(per["2026A"]["cfo"])
    assert rows["FY26"][3] == "148%"
    for token in ("43%", "148%", "67%", "rose when revenue fell"):
        assert token in c["alt"], token
    assert c["static_note"] is None                          # capex is separable in this data


def test_capex_flag_goes_in_static_note_when_unavailable(ctx):
    bad = copy.deepcopy(ctx)
    bad["extras"]["cashflow"]["capex_available"] = False
    out = charts.build_panel_charts(bad, FIX)
    note = _chart(out, "cash_conversion")["static_note"]
    assert "Capex cannot be separated" in note and "not presented as capex" in note


def test_debt_and_coverage_uses_ratio_bar_scales(panels, ctx):
    c = _chart(panels, "debt_and_coverage")
    rows = {r[0]: r for r in c["datatable"]["rows"]}
    assert rows["Debt to equity"][1] == "0.37x" and rows["Debt to equity"][2] == "0.37x to 0.80x" and rows["Debt to equity"][3] == "0.18x"
    assert rows["Interest coverage"][1] == "2.78x" and rows["Interest coverage"][2] == "2.78x to 7.65x"
    titles = [t["text"] for t in c["option"]["title"]]
    assert "0.80x in FY22 to 0.37x in FY26" in titles[0] and "lower is better" in titles[0]
    assert "higher is better" in titles[1]


# ---------------------------------------------------------------- panel 5
def test_valuation_strip_snapshot_old_multiple_and_toggle(panels, ctx):
    c = _chart(panels, "valuation_strip")
    assert [v["key"] for v in c["variants"]] == ["today", "report"] and c["default_variant"] == "today"
    assert c["option"] == c["variants"][0]["option"]
    today = c["variants"][0]["option"]["series"]
    subject = next(s for s in today if s["name"] == "WELSPUNLIV")["data"][0]
    assert subject["value"][0] == pytest.approx(22.02, abs=0.01)
    faded = next(s for s in today if s["name"] == "Old report multiple")["data"][0]
    assert faded["value"][0] == pytest.approx(16.59, abs=0.01)
    assert next(s for s in today if s["name"] == "Old report multiple")["itemStyle"]["opacity"] < 0.5   # shown faded
    report = c["variants"][1]["option"]["series"]
    assert next(s for s in report if s["name"] == "WELSPUNLIV")["data"][0]["value"][0] == pytest.approx(16.59, abs=0.01)
    assert "about Rs 129" in c["static_note"] and "16.6x" in c["static_note"]
    for token in ("Rs 249.00", "22.0x", "19.9x", "16.6x", "9.5x", "24.5x"):
        assert token in c["alt"], token
    median = next(s for s in today if s["name"] == "Peers")["markLine"]["data"][0]["xAxis"]
    assert median == pytest.approx(ctx["peers"]["current"]["median_excl_subject_ttm"], abs=0.01)


def test_pe_band_uses_earlier_years_and_two_dots_at_today_price(panels, ctx):
    c = _chart(panels, "pe_vs_history")
    rows = {r[0]: r for r in c["datatable"]["rows"]}
    assert rows["Own range low"][2] == "17.7x" and rows["Own range high"][2] == "36.9x"     # FY22 to FY25, latest year excluded
    assert rows["P/E on TTM EPS, today's price"][2] == "54.5x" and rows["P/E on FY26 EPS, today's price"][2] == "116.9x"
    for token in ("54.5x", "116.9x", "17.7x", "36.9x"):
        assert token in c["alt"]


def test_price_ruler_per_share_reconciles_to_the_fair_value_grid(panels, ctx):
    c = _chart(panels, "price_ruler")
    grid = {g["multiple"]: g["implied_price"] for g in ctx["fair_value"]["grid"]}
    assert charts.per_share(12, ctx["fair_value"]["ebitda_cr"], ctx["balance_latest"]["net_debt_cr"], ctx["market"]["shares_mn"]) == pytest.approx(grid[12], abs=0.01)
    rows = c["datatable"]["rows"]
    assert rows[0][1] == cc.fmt_rs(grid[12], 0) and rows[0][2] == cc.fmt_rs(grid[18], 0)
    assert any("not consensus" in r[0] for r in rows)        # no analyst consensus exists; the projection is labelled as such
    line = next(s for s in c["option"]["series"] if "markLine" in s)["markLine"]["data"][0]["xAxis"]
    assert line == pytest.approx(249.0)
    assert "Rs 249.00" in c["alt"]


# ---------------------------------------------------------------- panel 6
def test_price_trend_variants_chronological_crossover_and_volume(panels, ctx):
    c = _chart(panels, "price_trend")
    assert [v["key"] for v in c["variants"]] == ["6m", "1y", "2y"] and c["default_variant"] == "1y"
    assert c["option"] == c["variants"][1]["option"]
    sizes = {v["key"]: len(v["option"]["series"][0]["data"]) for v in c["variants"]}
    assert sizes["6m"] == 126 and sizes["1y"] == 250 and sizes["2y"] > sizes["1y"]
    for v in c["variants"]:
        dates = [d[0] for d in v["option"]["series"][0]["data"]]
        assert dates == sorted(dates)
        assert len(v["option"]["grid"]) == 2 and v["option"]["series"][3]["type"] == "bar"      # volume beneath
    pts = [p["label"]["formatter"] for p in c["option"]["series"][0]["markPoint"]["data"]]
    assert any(p.startswith("Golden cross") for p in pts) and any(p.startswith("Latest high") for p in pts)
    assert any(p.startswith("Trend start") for p in pts)
    assert c["datatable"]["rows"][-1][0] == "2026-11-13" and c["datatable"]["rows"][-1][1] == "249.00"
    for token in ("Rs 249.00", "13 Nov 26", "200-day", "1 Jun 26", "0.24x"):
        assert token in c["alt"], token
    assert "6-month" in c["static_note"]


def test_momentum_marks_the_bearish_divergence_from_real_data(panels, ctx):
    c = _chart(panels, "momentum")
    macd = next(s for s in c["option"]["series"] if s["name"] == "MACD")
    line = macd["markLine"]["data"][0]
    assert line[0]["coord"] == ["2026-10-06", 11.77] and line[1]["coord"][1] == pytest.approx(9.07)
    assert line[1]["label"]["formatter"] == "MACD 11.8 to 9.1"
    rsi = next(s for s in c["option"]["series"] if s["name"] == "RSI (14)")
    assert any(isinstance(d, list) and d[0].get("coord", [None, None])[1] == 82.11 for d in rsi["markLine"]["data"])
    for token in ("higher high", "lower high", "11.77", "9.07", "bearish divergence", "58.9"):
        assert token in c["alt"], token


def test_no_divergence_text_when_none_detected(ctx):
    flat = copy.deepcopy(ctx)
    flat["extras"]["technical"]["divergence"] = {"bearish": [], "bullish": []}
    # the chart recomputes from the price file, so use a short synthetic series without a swing pair
    import tempfile
    n = 300
    prices = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n), "open": 100.0, "high": [100 + i * 0.1 for i in range(n)],
                           "low": [99 + i * 0.1 for i in range(n)], "close": [100 + i * 0.1 for i in range(n)], "volume": 1000.0})
    with tempfile.TemporaryDirectory() as d:
        prices.to_csv(os.path.join(d, "historical_price_full.csv"), index=False)
        out = charts.build_panel_charts(flat, d)
    m = _chart(out, "momentum")
    assert "No divergence" in m["alt"]
    assert all("markLine" not in s or s["name"] == "RSI (14)" for s in m["option"]["series"])


# ---------------------------------------------------------------- panel 1
def test_price_vs_earnings_strip_matches_the_technical_facts(panels, ctx):
    c = _chart(panels, "price_vs_earnings")
    pve = ctx["extras"]["technical"]["price_vs_earnings"]
    assert c["datatable"]["rows"][0][1] == f"{pve['price_change_1y_pct']:+.1f}%"
    assert "+82%" in c["alt"] and "-68%" in c["alt"] and "+115%" in c["alt"]


# ---------------------------------------------------------------- graceful degradation
def test_bank_kind_returns_every_panel_with_no_charts(ctx):
    out = charts.build_panel_charts(ctx, FIX, kind="bank")
    assert list(out) == charts.PANEL_IDS and all(v == [] for v in out.values())


def test_missing_inputs_do_not_raise_and_are_recorded(tmp_path, ctx):
    out = charts.build_panel_charts(ctx, str(tmp_path))
    assert out["profit_engine"] == [] and out["cyclicality"] == [] and out["price"] == []
    assert out["valuation"] and out["strength"]                      # context-only charts still build
    assert any("financial_metrics_and_forecasts.csv" in e for e in charts.LAST_ERRORS)
    empty = charts.build_panel_charts({"meta": {}, "market": {}}, str(tmp_path))
    assert list(empty) == charts.PANEL_IDS and all(v == [] for v in empty.values())
    assert charts.LAST_ERRORS


def test_datatable_cells_are_strings_and_widths_match(panels):
    for _, c in _all(panels):
        width = len(c["datatable"]["columns"])
        assert all(len(r) == width and all(isinstance(x, str) for x in r) for r in c["datatable"]["rows"]), c["id"]
