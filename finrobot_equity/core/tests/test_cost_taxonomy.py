"""Cost taxonomy: industrial vs generic labels, the split cost rows, EBITDA definition and the end-to-end label pass."""
import copy
import json
import os
import shutil
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from modules import cost_taxonomy as ct  # noqa: E402
from modules import report_context as rc  # noqa: E402
from modules.finedge_api import normalize_income_statement  # noqa: E402
from modules.text_generator_agents import get_system_prompt  # noqa: E402
from report_visual import model as M  # noqa: E402
from test_forecast_caps import CFG, mahamaya  # noqa: E402
from modules.financial_data_processor import calculate_growth_and_forecasts  # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "visual_welspun")


@pytest.mark.parametrize("sector,kind", [
    ("Iron & Steel Products", "industrial"), ("Cement & Cement Products", "industrial"), ("Auto Components", "industrial"),
    ("Chemicals", "industrial"), ("Power Generation", "industrial"), ("Paper & Paper Products", "industrial"),
    ("Sugar", "industrial"), ("Oil Exploration", "industrial"),
    ("Other Textile Products", "generic"), ("Cigarettes & Tobacco", "generic"), ("Computers - Software", "generic"),
    ("Pharmaceuticals", "generic"), ("Retailing", "generic"),
])
def test_sector_classification(sector, kind):
    assert ct.resolve(sector)["kind"] == kind


def test_unknown_sector_is_generic_with_a_stated_limitation():
    t = ct.resolve("Quantum Widgets")
    assert t["kind"] == "generic" and t["matched"] is None and "not classified" in t["limitation"]
    assert ct.resolve("Other Textile Products")["limitation"] is None
    assert ct.resolve(kind="generic")["limitation"] is None                    # banks and explicit generic: no warning


def test_macro_sector_and_industry_are_used_when_the_sector_is_silent():
    assert ct.resolve("Specialty Items", "Industrial Products", "Industrials")["kind"] == "industrial"


def test_rename_rows_handles_the_crore_suffix_and_generic_is_unchanged():
    df = pd.DataFrame({"metrics": ["Revenue (INR crore)", "Cost of Operations (INR crore)", "SG&A (INR crore)", "Contribution Margin",
                                   "EBITDA (INR crore)"], "2026A": [1, 2, 3, "20.8%", 4]})
    ind = ct.rename_rows(df, ct.resolve("Iron & Steel Products"))
    assert list(ind["metrics"]) == ["Revenue (INR crore)", "Materials cost (consumed + purchases + change in inventory) (INR crore)",
                                    "Employee and other operating costs (INR crore)", "Material margin (% of revenue)", "EBITDA (INR crore)"]
    assert list(ct.rename_rows(df, ct.resolve("Other Textile Products"))["metrics"]) == list(df["metrics"])


def test_apply_text_relabels_prose_and_leaves_identifiers_and_generic_alone():
    t = ct.resolve("Iron & Steel Products")
    out = ct.apply_text({"id": "contribution_margin", "label": "Contribution margin", "so_what": "Pricing is holding; any profit squeeze is further down.",
                         "text": "EBITDA moved: contribution margin +2.5 points, SG&A as % of revenue +3.0 points.",
                         "glossary": [{"term": "SG&A"}]}, t)
    assert out["id"] == "contribution_margin" and out["label"] == "Material margin"
    assert out["so_what"].startswith("The spread over materials is holding")
    assert "material margin +2.5" in out["text"] and "employee and other operating costs as % of revenue +3.0" in out["text"]
    assert out["glossary"] == [{"term": "SG&A"}]                                # glossary handled separately
    g = ct.resolve("Other Textile Products")
    assert ct.apply_text("Contribution margin and SG&A", g) == "Contribution margin and SG&A"


def test_split_rows_reconcile_to_the_sga_bucket_in_every_year_including_forecasts():
    final = calculate_growth_and_forecasts(mahamaya(), {**CFG})
    income = pd.DataFrame({"year": [2022, 2023, 2024, 2025, 2026],
                           "employeeBenefitExpense": [50.647e6, 67.417e6, 98.812e6, 141.309e6, 227.723e6],
                           "otherExpenses": [700.322e6, 923.909e6, 1105.028e6, 1175.554e6, 1373.856e6]})
    out = ct.add_split_rows(final, income, ct.resolve("Iron & Steel Products"))
    names = list(out["metrics"])
    assert names.index("Employee Cost") == names.index("SG&A") + 1 and "Other Operating Expenses" in names
    assert out.attrs["margin_cap"] == final.attrs["margin_cap"]                  # attrs survive
    for y in [c for c in out.columns if isinstance(c, str) and c[:4].isdigit()]:
        e = float(out.loc[out["metrics"] == "Employee Cost", y].iloc[0])
        o = float(out.loc[out["metrics"] == "Other Operating Expenses", y].iloc[0])
        s = float(out.loc[out["metrics"] == "SG&A", y].iloc[0])
        assert e + o == pytest.approx(s, rel=1e-6, abs=5)
    # generic companies get no split rows
    assert "Employee Cost" not in list(ct.add_split_rows(final, income, ct.resolve("Other Textile Products"))["metrics"])


def test_ebitda_is_revenue_less_total_operating_expenses_and_excludes_other_income():
    row = {"period_end": 20260331, "year": 2026, "revenueFromOperations": 8828.483e6, "expenses": 8730.218e6, "financeCosts": 50.166e6,
           "depreciationAndAmortisation": 88.122e6, "costOfMaterialsConsumed": 6500e6, "purchasesOfStockInTrade": 300e6,
           "changesInInventories": 190.351e6, "employeeBenefitExpense": 227.723e6, "otherExpenses": 1373.856e6,
           "otherIncome": 15.757e6, "exceptionalItemsBeforeTax": -9e6, "profitBeforeTax": 100e6, "taxExpense": 20e6,
           "profitOrLossAttributableToOwners": 80e6}
    inc = normalize_income_statement([row], "annual").iloc[0]
    assert inc["ebitda"] == pytest.approx(8828.483e6 - (8730.218e6 - 50.166e6 - 88.122e6))
    assert inc["ebitda"] == pytest.approx(236.553e6, abs=1e3)
    row2 = dict(row, otherIncome=500e6, exceptionalItemsBeforeTax=-300e6)
    assert normalize_income_statement([row2], "annual").iloc[0]["ebitda"] == pytest.approx(inc["ebitda"])     # other income / exceptionals excluded
    assert "revenue from operations" in ct.EBITDA_DEFINITION and "Other income" in ct.EBITDA_DEFINITION
    # materials cost = consumed + purchases + change in inventory (costOfRevenue)
    assert inc["costOfRevenue"] == pytest.approx(6500e6 + 300e6 + 190.351e6)


def test_taxonomy_detected_from_a_reloaded_table():
    final = calculate_growth_and_forecasts(mahamaya(), {**CFG})
    income = pd.DataFrame({"year": [2022, 2023, 2024, 2025, 2026], "employeeBenefitExpense": [1e6] * 5, "otherExpenses": [2e6] * 5})
    assert ct.taxonomy_from_table(final) is None
    assert ct.is_industrial(ct.taxonomy_from_table(ct.add_split_rows(final, income, ct.resolve("Iron & Steel Products"))))


def test_industrial_prompt_uses_material_margin_labels_and_generic_prompt_is_unchanged():
    ind = get_system_prompt("major_takeaways", False, "industrial")
    assert "Material Margin:" in ind and "Operating Cost Ratio:" in ind and "SG&A" not in ind.split("WRITING RULES")[0]
    gen = get_system_prompt("major_takeaways", False, None)
    assert "SG&A Expense Margin" in gen and get_system_prompt("major_takeaways", False, "generic") == gen
    assert "Net Interest" in get_system_prompt("major_takeaways", True, "industrial")       # banks keep their own prompt


def _industrial_fixture(tmp_path):
    d = tmp_path / "ind"
    shutil.copytree(FIX, d)
    ctx = json.load(open(d / "report_context.json"))
    ctx["taxonomy"] = ct.resolve("Iron & Steel Products")
    ctx["forecast_caps"] = {"cap": 3.55, "peak": 3.12, "std": 0.43, "binding": True, "lifted": False, "trigger": None,
                            "message": "EBITDA margin forecast capped at 3.6% = historical peak 3.1% + 1 sd (0.4 pts); no fundamental trigger supplied."}
    json.dump(ctx, open(d / "report_context.json", "w"))
    return str(d)


def test_visual_model_carries_the_taxonomy_flags_and_relabelled_text(tmp_path):
    m = M.build_report_model(_industrial_fixture(tmp_path), charts_fn=lambda *a, **k: {})
    assert m["meta"]["taxonomy"]["kind"] == "industrial" and m["meta"]["forecast_caps"]["binding"] is True
    texts = json.dumps({k: v for k, v in m.items() if k != "glossary"})
    assert "SG&A" not in texts and "Contribution margin" not in texts and "contribution margin" not in texts
    labels = [c["label"] for p in m["panels"] for c in p["chips"]]
    assert "Material margin" in labels and "Employee and other operating costs as % of revenue" in labels
    assert next(p for p in m["panels"] if p["id"] == "profit_engine")["headline"].startswith("Spreads held")
    msgs = [w["text"] for w in m["warnings"]]
    assert any("capped at 3.6%" in t for t in msgs) and any("Power and fuel are inside other operating expenses" in t for t in msgs)
    terms = {g["id"]: g for g in m["glossary"]}
    assert terms["contribution_margin"]["term"] == "Material margin" and "Ind-AS" in terms["ebitda"]["definition"]
    assert m["sources"]["validation"]["errors"] == []


def test_generic_company_keeps_its_labels(tmp_path):
    m = M.build_report_model(FIX, charts_fn=lambda *a, **k: {})
    labels = [c["label"] for p in m["panels"] for c in p["chips"]]
    assert "Contribution margin" in labels and "SG&A as % of revenue" in labels
    assert m["panels"][1]["headline"] == "Prices held. Costs broke the margin."


def test_text_data_tables_use_taxonomy_names_and_notes():
    from test_report_context import _ctx, _final_df
    ctx = _ctx()
    ctx["taxonomy"] = ct.resolve("Iron & Steel Products")
    ctx["forecast_caps"] = {"message": "EBITDA margin forecast capped at 3.6% = historical peak 3.1% + 1 sd (0.4 pts); no fundamental trigger supplied."}
    data = rc.build_text_data(ctx, _final_df(), None, None, base={"data_notes": []})
    names = list(data["financial_metrics"]["metrics"])
    assert "Materials cost (consumed + purchases + change in inventory) (INR crore)" in names and "Material margin (INR crore)" in names
    assert data["taxonomy"]["kind"] == "industrial"
    assert any("capped at 3.6%" in n for n in data["data_notes"]) and any("power and fuel" in n.lower() for n in data["data_notes"])
