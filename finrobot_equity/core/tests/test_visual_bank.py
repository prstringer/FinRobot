"""Tests for the bank variant of the visual report (config-driven panels, bank verdicts, context, charts, validation)."""
import copy
import json
import os
import re
import statistics
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from report_visual import validate as V  # noqa: E402
from report_visual import verdict_engine as ve  # noqa: E402
from report_visual.bank import build as B  # noqa: E402
from report_visual.bank import charts as BC  # noqa: E402
from report_visual.bank import context as BX  # noqa: E402
from report_visual.bank import headlines as BH  # noqa: E402
from report_visual.bank import providers as P  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "visual_bank")


@pytest.fixture(scope="module")
def model():
    return B.build_bank_report_model(FIX)


@pytest.fixture(scope="module")
def ctx():
    return json.load(open(os.path.join(FIX, "report_context.json"), encoding="utf-8"))


def chips(model):
    return {c["id"]: c for p in model["panels"] for c in p["chips"]}


def test_seven_panels_bank_kind_and_short_headlines(model):
    assert model["meta"]["kind"] == "bank"
    assert [p["id"] for p in model["panels"]] == V.PANEL_IDS
    for p in model["panels"]:
        assert 0 < len(re.findall(r"[\w']+", p["headline"])) <= 8, p["headline"]


def test_every_chip_is_complete_and_so_what_does_not_restate_the_value(model):
    cs = chips(model)
    assert {"nim", "cost_to_income", "credit_cost", "gnpa", "cet1", "bank_roe", "pb", "pe", "fair_value"} <= set(cs)
    for c in cs.values():
        assert c["verdict"] in ("good", "neutral", "concern") and c["icon"] in ("check", "dash", "alert")
        assert c["yardstick"] in ("own_history", "peers", "general_norm")
        assert c["scale"]["min"] is not None and c["scale"]["max"] is not None and c["calc"]
        assert len(c["so_what"].split()) <= 30 and c["value_text"] not in c["so_what"]
        if c["yardstick"] == "general_norm":
            assert c["yardstick_text"] == "General norm, not company-specific"


def test_no_ebitda_measure_anywhere_in_the_bank_model(model):
    blob = json.dumps(model, ensure_ascii=False).upper()
    assert "EBITDA" not in blob and "CONTRIBUTION MARGIN" not in blob and "SG&A" not in blob


def test_validation_has_no_errors_and_fair_value_cases_recompute(model):
    v = model["sources"]["validation"]
    assert v["errors"] == []
    assert any("fair-value cases recompute" in p for p in v["passed"])
    assert not any(w["level"] == "error" for w in model["warnings"])


def test_fair_value_per_share_against_the_snapshot_price(model, ctx):
    price = model["meta"]["snapshot"]["price"]
    val = next(p for p in model["panels"] if p["id"] == "valuation")
    cases = val["extra"]["fair_value_cases"]
    assert len(cases) >= 3 and val["extra"]["price"] == price
    pb_hist = [r["pb"] for r in ctx["fy_series"].values() if r.get("pb")]
    own = next(c for c in cases if c["key"] == "pb_own_history")
    assert own["low_price"] == pytest.approx(min(pb_hist) * ctx["balance_latest"]["bvps"], rel=1e-9)
    assert own["high_price"] == pytest.approx(max(pb_hist) * ctx["balance_latest"]["bvps"], rel=1e-9)
    for c in cases:
        assert c["low_price"] == pytest.approx(c["low_mult"] * c["base_value"], rel=1e-9)
        assert c["high_price"] == pytest.approx(c["high_mult"] * c["base_value"], rel=1e-9)
        assert c["vs_price_pct_low"] == pytest.approx((c["low_price"] / price - 1) * 100, abs=0.01)
    assert chips(model)["fair_value"]["scale"]["value"] == price


def test_peer_medians_exclude_the_subject(ctx):
    pb = ctx["peers"]["bank"]
    others = [r["pb_now"] for t, r in pb["rows"].items() if t != "HDFCBANK"]
    assert pb["median_excl_subject"]["pb_now"] == pytest.approx(statistics.median(others))
    assert "EXCLUDE" in pb["policy"] and pb["n_peers"] == 5


def test_bank_verdicts_come_from_the_merged_config_with_direction():
    cfg = ve.load_config()["metrics"]
    assert cfg["cost_to_income"]["higher_is_better"] is False and cfg["gnpa"]["higher_is_better"] is False
    assert cfg["nim"]["higher_is_better"] is True and cfg["cet1"]["general_norm"] == {"good": 12.0, "concern": 9.0}
    hist = [10, 12, 14, 16, 18]
    assert ve.evaluate("nim", 18, hist)["verdict"] == "good"
    assert ve.evaluate("cost_to_income", 18, hist)["verdict"] == "concern"
    assert ve.evaluate("gnpa", 10, hist)["verdict"] == "good"
    assert ve.evaluate("cet1", 12.5, [])["yardstick"] == "general_norm"           # fallback is labelled


def test_merger_year_is_left_out_of_the_loan_growth_range(model):
    c = chips(model)["advances_growth"]
    assert "FY24" in c["calc"] and "merger" in c["calc"]
    assert c["scale"]["max"] < 40                                                  # the 55% year is not the top of the range
    assert any("merger" in w["text"].lower() for w in model["warnings"])


def test_panels_are_driven_by_config_not_code(monkeypatch):
    variant = copy.deepcopy(P.load_variant())
    variant["panels"]["profit_engine"]["chips"] = ["cost_to_income", "credit_cost"]
    variant["panels"]["profit_engine"]["charts"] = ["income_split"]
    monkeypatch.setattr(P, "load_variant", lambda path=None: variant)
    m = B.build_bank_report_model(FIX)
    p = next(p for p in m["panels"] if p["id"] == "profit_engine")
    assert [c["id"] for c in p["chips"]] == ["cost_to_income", "credit_cost"]
    assert [c["id"] for c in p["charts"]] == ["income_split"]


def test_signposts_come_from_own_history(model, ctx):
    sp = next(p for p in model["panels"] if p["id"] == "watch")["extra"]["signposts"]
    assert 3 <= len(sp) <= 5
    nim = next(s for s in sp if s["metric"] == "Net interest margin")
    hist = list(P.actual_series(ctx, "NIM").values())
    assert nim["threshold"] == pytest.approx(round(statistics.median(hist) / 0.1) * 0.1) and nim["threshold_text"].startswith("Above")
    gnpa = next(s for s in sp if s["metric"] == "Gross NPA")
    assert gnpa["threshold_text"].startswith("Below")


def test_headlines_cover_branches_and_stay_short():
    chips_ = {"nim": {"verdict": "concern"}, "cost_to_income": {"verdict": "concern"}}
    assert BH.profit_engine(chips_, {}) == "Margin and costs both slipping."
    assert BH.cycle({}, {"downturn_years": 2, "growth_observations": 4}) == "Second profit dip in four years."
    assert BH.valuation({}, {"pb_vs_history": "within", "pb_vs_peers": "within"}) == "Priced near its history and level with peers."
    for rule, fn in BH.RULES.items():
        for a in ("good", "neutral", "concern"):
            for b in ("good", "neutral", "concern"):
                ch = {k: {"verdict": a} for k in ("nim", "gnpa")} | {k: {"verdict": b} for k in ("cost_to_income", "cet1", "credit_cost")}
                h = fn(ch, {"downturn_years": 1, "growth_observations": 4, "pb_vs_history": "above", "pb_vs_peers": "above"})
                assert len(re.findall(r"[\w']+", h)) <= 8, h


def test_charts_are_valid_chronological_and_traceable(ctx):
    variant = P.load_variant()
    facts = P.compute_facts(ctx, variant)
    cases = P.fair_value_cases(ctx, variant, facts)
    panels = BC.build_bank_panel_charts(ctx, FIX, variant, cases, facts["price"])
    ids = {pid: [c["id"] for c in cs] for pid, cs in panels.items()}
    assert ids["profit_engine"] == ["income_split", "nim_trend"] and ids["valuation"] == ["pb_peer_strip", "fair_value_ruler", "pb_vs_roe"]
    assert ids["strength"] == ["asset_quality", "capital_returns"] and "price_trend" in ids["price"]
    for pid, cs in panels.items():
        mains = [c for c in cs if c["role"] == "main"]
        assert len(mains) <= 1 and len([c for c in cs if c["role"] == "small"]) <= 2, pid
        for c in cs:
            assert c["alt"] and c["datatable"]["rows"] and c["option"]["animation"] is False
            assert not re.search(r'"#[0-9a-fA-F]{3,8}"', json.dumps(c["option"]))
    split = next(c for c in panels["profit_engine"] if c["id"] == "income_split")
    years = split["option"]["xAxis"]["data"]
    assert years == sorted(years)                                                   # oldest to newest, left to right
    for row in split["datatable"]["rows"]:
        assert sum(float(x) for x in row[1:5]) == pytest.approx(100.0, abs=0.3)      # every bar sums to Rs 100 of income
    strip = next(c for c in panels["valuation"] if c["id"] == "pb_peer_strip")
    assert [v["key"] for v in strip["variants"]] == ["today", "report"] and "P/B was" in strip["static_note"]


def test_injected_inconsistency_is_flagged(model):
    bad = copy.deepcopy(model)
    bad["meta"]["snapshot"]["price"] = 1000.0
    res = V.validate_model(bad)
    assert res["errors"]


def test_fair_value_check_catches_a_wrong_case(model):
    val = next(p for p in model["panels"] if p["id"] == "valuation")
    cases = copy.deepcopy(val["extra"]["fair_value_cases"])
    cases[0]["low_price"] *= 1.5
    assert P.check_cases(cases, model["meta"]["snapshot"]["price"])


def test_bank_context_parses_the_table_and_excludes_subject_from_medians():
    df = pd.read_csv(os.path.join(FIX, "financial_metrics_and_forecasts.csv"))
    s = BX.bank_series(df)
    assert s["NIM"]["2026A"] == pytest.approx(3.12) and s["Net Profit Growth"]["2024A"] == pytest.approx(37.9)
    peers = pd.read_csv(os.path.join(FIX, "peer_bank_comparison.csv"))
    block = BX._peer_block(peers, {"HDFCBANK": {"pb_now": 2.0, "price": 1.0}, "ICICIBANK": {"pb_now": 3.0, "price": 1.0}}, "HDFCBANK")
    others = [r["roe"] for t, r in block["rows"].items() if t != "HDFCBANK"]
    assert block["median_excl_subject"]["roe"] == pytest.approx(statistics.median(others))
    assert block["rows"]["HDFCBANK"]["roe"] == pytest.approx(13.37)


def test_context_keeps_skeleton_keys_so_existing_consumers_do_not_break(ctx):
    for k in ("meta", "market", "balance_latest", "current_valuation", "ttm", "ratios", "latest_fy", "fy_series", "model_forecast",
              "peers", "fair_value", "derived", "technicals", "insights", "extras", "checks", "prompt_block"):
        assert k in ctx, k
    assert ctx["current_valuation"]["ev_cr"] is None and ctx["fair_value"]["grid"] == []
    assert ctx["current_valuation"]["pe_ttm_computed"] == pytest.approx(ctx["market"]["price"] / ctx["ttm"]["eps"], rel=1e-6)


def test_renders_to_html_when_the_renderer_is_available(model, tmp_path):
    render_html = pytest.importorskip("report_visual.render_html")
    out = render_html.render_html(model, str(tmp_path / "bank.html"))
    text = open(out, encoding="utf-8").read()
    assert os.path.getsize(out) < 2 * 1024 * 1024
    for p in model["panels"]:
        assert p["headline"] in text
