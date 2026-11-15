"""Tests for the visual report model: verdict engine, headlines, fair value, signposts, validation, CLI wiring."""
import copy
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from report_visual import headlines as hl  # noqa: E402
from report_visual import model as M  # noqa: E402
from report_visual import validate as V  # noqa: E402
from report_visual import verdict_engine as ve  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "visual_welspun")


@pytest.fixture(scope="module")
def model():
    return M.build_report_model(FIX, charts_fn=lambda *a, **k: {})


def chips(model):
    return {c["id"]: c for p in model["panels"] for c in p["chips"]}


def test_seven_panels_in_order_with_short_conclusion_headlines(model):
    assert [p["id"] for p in model["panels"]] == V.PANEL_IDS
    for p in model["panels"]:
        assert 0 < len(re.findall(r"[\w']+", p["headline"])) <= 8, p["headline"]
    heads = {p["id"]: p["headline"] for p in model["panels"]}
    assert heads["profit_engine"] == "Prices held. Costs broke the margin."
    assert heads["cyclicality"] == "Second downturn in four years, same pattern."
    assert heads["strength"] == "Debt falling; cash strong, partly one-off."
    assert heads["price"] == "Strong trend, losing force."


def test_thesis_matches_the_reference_case(model):
    assert model["thesis"] == "A sound product business, hurt by fixed costs, priced for a recovery that has only started."


def test_every_chip_has_yardstick_scale_and_a_short_so_what(model):
    cs = chips(model)
    assert len(cs) >= 9
    for c in cs.values():
        assert c["verdict"] in ("good", "neutral", "concern") and c["icon"] in ("check", "dash", "alert")
        assert c["yardstick"] in ("own_history", "peers", "general_norm")
        assert c["scale"]["min"] is not None and c["scale"]["max"] is not None
        assert len(c["so_what"].split()) <= 30 and c["value_text"] not in c["so_what"]
        assert c["source"] in ("report_data", "calculated", "general_norm")
        assert c["calc"]
        if c["yardstick"] == "general_norm":
            assert c["yardstick_text"] == "General norm, not company-specific"


def test_reference_verdicts_for_welspunliv(model):
    cs = chips(model)
    assert cs["contribution_margin"]["verdict"] == "good"
    assert cs["sga_pct"]["verdict"] == "concern" and cs["sga_pct"]["yardstick"] == "own_history"
    assert cs["debt_equity"]["verdict"] == "good"                     # 0.80 -> 0.37, improving
    assert cs["interest_coverage"]["verdict"] == "concern"
    assert cs["cash_conversion"]["verdict"] == "concern"              # flattered by the cyclical pattern
    assert cs["operating_leverage"]["value"] == pytest.approx(44.1, abs=0.2)
    assert cs["fair_value"]["verdict"] == "concern"                   # price above every case


def test_direction_is_encoded_per_metric():
    hist = [10, 12, 14, 16, 18]
    assert ve.evaluate("contribution_margin", 18, hist)["verdict"] == "good"
    assert ve.evaluate("sga_pct", 18, hist)["verdict"] == "concern"
    assert ve.evaluate("debt_equity", 18, hist)["verdict"] == "concern"
    assert ve.evaluate("interest_coverage", 18, hist)["verdict"] == "good"


def test_no_verdict_without_a_yardstick():
    assert ve.evaluate("contribution_margin", 40.0, [40.0]) is None
    assert ve.evaluate("roe", 10.0, []) is not None                    # general norm fallback
    assert ve.evaluate("roe", 10.0, [])["yardstick"] == "general_norm"
    assert ve.evaluate("roe", None, [1, 2, 3]) is None


def test_thresholds_live_in_the_config_not_the_code():
    cfg = ve.load_config()
    assert cfg["defaults"]["own_history"]["good_pos"] == 0.65
    assert cfg["metrics"]["sga_pct"]["higher_is_better"] is False
    assert cfg["metrics"]["interest_coverage"]["general_norm"] == {"good": 4.0, "concern": 2.0}


def test_margin_bridge_and_operating_leverage_reach_the_model(model):
    p = next(p for p in model["panels"] if p["id"] == "profit_engine")
    b = p["extra"]["bridge"]
    assert b["d_sga_pct_pts"] == pytest.approx(7.1, abs=0.06)
    assert b["d_ebitda_margin_pts"] == pytest.approx(-6.2, abs=0.06)
    assert b["d_contribution_margin_pts"] == pytest.approx(0.9, abs=0.06)
    assert p["extra"]["operating_leverage"]["ebitda_per_100_revenue"] == pytest.approx(44.1, abs=0.2)


def test_valuation_uses_the_snapshot_price_and_per_share_cases(model):
    snap = model["meta"]["snapshot"]
    assert snap["price"] == 249.0 and snap["price_date"] == "2026-11-13"
    val = next(p for p in model["panels"] if p["id"] == "valuation")
    cases = val["extra"]["fair_value_cases"]
    assert len(cases) >= 3 and val["extra"]["price"] == 249.0
    for c in cases:
        assert c["low_price"] <= c["high_price"]
        assert c["vs_price_pct_low"] == pytest.approx((c["low_price"] / 249.0 - 1) * 100, abs=0.01)
    assert all(c["high_price"] < 249.0 for c in cases)               # priced above every case
    assert "16." in (val["extra"]["toggle_note"] or "")               # the old report's multiple is kept, labelled


def test_cases_say_no_consensus_is_available(model):
    assert any("consensus" in w["text"].lower() for w in model["warnings"])
    assert not any("consensus" in c["key"] for p in model["panels"] for c in p["extra"].get("fair_value_cases", []))


def test_signposts_come_from_history_and_forecasts_and_portfolio_is_empty(model):
    w = next(p for p in model["panels"] if p["id"] == "watch")
    sp = w["extra"]["signposts"]
    assert 3 <= len(sp) <= 5
    assert any(s["metric"] == "SG&A as % of revenue" and "34" in s["threshold_text"] for s in sp)
    assert all(s["status"] in ("good", "neutral", "concern") for s in sp)
    assert w["extra"]["portfolio"] == {"current_weight": None, "target_weight": None, "triggers": None}


def test_technical_tests_are_four_independent_tiles_without_a_vote(model):
    price = next(p for p in model["panels"] if p["id"] == "price")
    tests = price["extra"]["tests"]
    assert [t["key"] for t in tests] == ["trend", "momentum", "participation", "stretch"]
    assert "overall" not in json.dumps(price["extra"]).lower().replace("overall_", "")
    assert price["extra"]["divergence"] and price["extra"]["rsi_zone"] == "neutral"


def test_validation_passes_on_the_reference_model(model):
    v = model["sources"]["validation"]
    assert v["errors"] == []
    assert any("reconciliation" in p for p in v["passed"])


def test_injected_inconsistency_is_flagged_on_panel_1():
    bad = M.build_report_model(FIX, charts_fn=lambda *a, **k: {})
    bad["meta"]["snapshot"]["price"] = 129.0                           # the stale price the old report used
    res = V.validate_model(bad)
    assert any("market cap" in e or "P/E" in e for e in res["errors"])


def test_text_numbers_must_come_from_the_model():
    m = M.build_report_model(FIX, charts_fn=lambda *a, **k: {})
    m["panels"][1]["chips"][0]["so_what"] = "Margin is 99.9 points above the norm."
    assert any("99.9" in w for w in V.validate_model(m)["warnings"])


def test_headline_over_eight_words_fails():
    m = M.build_report_model(FIX, charts_fn=lambda *a, **k: {})
    m["panels"][2]["headline"] = "This headline is far too long to be a conclusion"
    assert any("Headline" in e for e in V.validate_model(m)["errors"])


def test_model_is_json_serialisable_and_saved(tmp_path, model):
    path = M.save_report_model(model, str(tmp_path))
    assert path.endswith("WELSPUNLIV_2026-11-15.report.json")
    loaded = json.load(open(path, encoding="utf-8"))
    assert loaded["meta"]["ticker"] == "WELSPUNLIV" and len(loaded["panels"]) == 7


def test_no_em_dashes_or_entities_anywhere_in_the_model(model):
    blob = json.dumps(model, ensure_ascii=False)
    assert "—" not in blob
    assert not re.search(r"&(?!amp;|lt;|gt;|quot;|#\d+;)[A-Za-z]+;", blob)


def test_headline_functions_cover_the_other_branches():
    assert hl.profit_engine({"contribution_margin": {"verdict": "concern"}, "sga_pct": {"verdict": "concern"}}, {}) == "Margins squeezed from both sides."
    assert hl.cyclicality({}, {"downturn_years": 0}) .startswith("No downturn")
    assert hl.price_behaviour({"trend": "concern", "momentum": "concern"}, {}) == "Downtrend with no momentum."
    assert hl.valuation({}, {"ev_vs_history": "below", "ev_vs_peers": "within"}) == "Priced below its history and in line with peers."
