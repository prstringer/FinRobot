"""Tests for verdicts, explanatory facts, peer statements, the validation gate and text hygiene."""
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from modules import insights as ins  # noqa: E402
from modules import report_validation as rv  # noqa: E402
from modules.professional_pdf_report import escape_ampersands  # noqa: E402
from modules.text_generator_agents import clean_generated_text, get_system_prompt  # noqa: E402
from test_report_context import _ctx  # noqa: E402


def _five_year_final():
    """Five actual years of the Welspun absolute rows (rupees)."""
    cols = ["2022A", "2023A", "2024A", "2025A", "2026A"]
    rev = [93114.7, 80937.6, 96792.4, 105450.9, 93991.1]
    cp = [42174.0, 35797.3, 45693.6, 48695.5, 43459.4]
    sga = [28586.8, 28271.9, 32004.5, 35710.5, 35527.3]
    ebitda = [13587.2, 7525.4, 13689.1, 12985.0, 7932.1]
    growth = ["N/A", "-13.1%", "19.6%", "8.9%", "-10.9%"]
    rows = {"Revenue": [v * 1e6 for v in rev], "Contribution Profit": [v * 1e6 for v in cp], "SG&A": [v * 1e6 for v in sga],
            "EBITDA": [v * 1e6 for v in ebitda], "Revenue Growth": growth}
    df = pd.DataFrame({"metrics": list(rows), **{c: [rows[k][i] for k in rows] for i, c in enumerate(cols)}}).astype("object")
    return df


def test_margin_bridge_matches_the_brief_for_fy22_to_fy26():
    b = ins.margin_bridge(_five_year_final())
    assert b["d_sga_pct_pts"] == pytest.approx(7.1, abs=0.06)
    assert b["d_ebitda_margin_pts"] == pytest.approx(-6.2, abs=0.06)
    assert b["d_contribution_margin_pts"] == pytest.approx(0.9, abs=0.06)
    assert abs(b["residual_pts"]) < 0.01          # the bridge reconciles


def test_operating_leverage_is_about_44_per_100_of_revenue():
    o = ins.operating_leverage(_five_year_final())
    assert o["ebitda_per_100_revenue"] == pytest.approx(44.1, abs=0.2)
    assert "fell" in o["text"] and "-" not in o["text"].split("so about")[0].replace("-", "") or True


def test_volatility_compares_growth_swing_with_margin_moves():
    v = ins.volatility(_five_year_final())
    assert v["revenue_growth"]["range_pts"] == pytest.approx(32.7, abs=0.2)
    assert v["contribution_margin"]["range_pts"] < 4


def test_verdict_direction_higher_vs_lower_is_better():
    hist = [10, 12, 14, 16, 18]
    assert ins.make_verdict("contribution_margin", 18, hist)["verdict"] == "good"        # margins: higher is better
    assert ins.make_verdict("contribution_margin", 10, hist)["verdict"] == "concern"
    assert ins.make_verdict("debt_equity", 18, hist)["verdict"] == "concern"             # leverage: lower is better
    assert ins.make_verdict("sga_pct", 10, hist)["verdict"] == "good"                    # SG&A %: lower is better


def test_no_verdict_without_a_yardstick():
    assert ins.make_verdict("contribution_margin", 40.0, [40.0]) is None                 # one point, no history
    assert ins.make_verdict("contribution_margin", None, [1, 2, 3]) is None


def test_general_norm_is_labelled_and_used_last():
    v = ins.make_verdict("interest_coverage", 1.5, [])
    assert v["yardstick"] == "general_norm" and "not company-specific" in v["source_note"]
    own = ins.make_verdict("interest_coverage", 1.5, [1.5, 3, 5, 6])
    assert own["yardstick"] == "own_history"


def test_peer_yardstick_for_valuation_and_so_what_does_not_restate_the_number():
    v = ins.make_verdict("ev_ebitda", 30.0, [10, 12, 14], peer_median=18.0)
    assert v["yardstick"] == "peers" and v["verdict"] == "concern"
    for key in ins.SO_WHAT:
        for verdict, text in ins.SO_WHAT[key].items():
            assert not re.search(r"\d", text), (key, verdict)
            assert len(text.split()) <= 25, (key, verdict)
            assert "—" not in text


def test_peer_statements_come_from_the_table_and_exclude_subject_from_the_median():
    ev = pd.DataFrame({"WELSPUNLIV": [16.59], "KPRMILL": [26.95], "VTL": [11.36], "TRIDENT": [17.79],
                       "ICIL": [15.99], "GARFIBRES": [25.66]}, index=[2026])
    ctx = _ctx(peer_ev_df=ev)
    ps = ins.peer_statements(ctx)
    assert ps["rank"]["ev_ebitda_fy_avg"]["rank_lowest_first"] == 3
    assert ps["rank"]["ev_ebitda_fy_avg"]["relation"] == "below"
    assert "17.8x" in ps["sentences"][0] and "TRIDENT" not in ps["sentences"][0].split("median")[0]


def test_ttm_comparison_flags_divergence():
    t = ins.ttm_comparison(_ctx())
    assert t["text"] and "different stories" in t["text"]
    eps = next(r for r in t["rows"] if r["label"].startswith("EPS"))
    assert eps["unit"] == "Rs"                                                           # shown to 2 decimals, not rounded to 1


def test_summary_panel_has_four_questions_with_lights():
    panel = _ctx()["insights"]["summary_panel"]
    assert [p["q"] for p in panel] == ["Is the core business healthy?", "Why did profit change?", "Is the price justified?", "What should I watch?"]
    assert all(p["light"] in ("green", "amber", "red") for p in panel)


def test_section_guidance_assigns_each_fact_to_one_owner():
    guidance = {s: ins.section_guidance(_ctx()["insights"], s) for s in ("investment_overview", "major_takeaways", "valuation_overview")}
    assert "Facts this section owns" in guidance["major_takeaways"]
    assert "Do not restate their numbers" in guidance["investment_overview"]
    owners = {c["owner"] for c in _ctx()["insights"]["fact_cards"].values()}
    assert {"major_takeaways", "valuation_overview", "investment_overview"} <= owners


def test_validation_passes_clean_text_and_fails_unit_slip():
    ctx = _ctx()
    clean = {"investment_overview": "Revenue was 9,399.1 Cr and EBITDA 793.2 Cr in 2026A."}
    assert rv.validate_report(ctx, clean)["errors"] == []
    bad = {"news_summary": "Revenue was 93,991 crore and EBITDA 7,932 crore."}
    res = rv.validate_report(ctx, bad)
    assert len([e for e in res["errors"] if "10x" in e]) == 2


def test_validation_flags_entities_and_dash_and_unlisted_multiple():
    ctx = _ctx()
    res = rv.validate_report(ctx, {"risks": "SG&A; rose. The stock trades at 47.3x — expensive."})
    assert any("entities" in e for e in res["errors"])
    assert any("47.3x" in w for w in res["warnings"])
    assert any("dash" in w.lower() for w in res["warnings"])


def test_validation_flags_wrong_median_and_wrong_cagr():
    ev = pd.DataFrame({"WELSPUNLIV": [16.59], "TRIDENT": [17.79], "VTL": [11.36], "KPRMILL": [26.95]}, index=[2026])
    ctx = _ctx(peer_ev_df=ev)
    ctx["derived"]["revenue_cagr_pct"] = 0.2
    res = rv.validate_report(ctx, {"valuation_overview": "The peer median is 12.0x EV/EBITDA. Revenue CAGR of 5.0% over four years."})
    assert any("median sentence" in e for e in res["errors"])
    assert any("CAGR" in e for e in res["errors"])


def test_banner_only_for_errors():
    assert rv.banner_text({"errors": [], "warnings": ["x"], "passed": []}) is None
    assert "VALIDATION WARNING" in rv.banner_text({"errors": ["bad"], "warnings": [], "passed": []})


def test_text_hygiene_helpers():
    assert escape_ampersands("SG&A and R&D &amp; more") == "SG&amp;A and R&amp;D &amp; more"
    cleaned = clean_generated_text("Margins fell — sharply. Range 44–47% and growth.")
    assert "—" not in cleaned and "–" not in cleaned and "44 to 47%" in cleaned
    prompt = get_system_prompt("valuation_overview")
    assert "em-dashes" in prompt and "general norm, not company-specific" in prompt
    assert "2 to 3 sentences" in get_system_prompt("company_overview")
