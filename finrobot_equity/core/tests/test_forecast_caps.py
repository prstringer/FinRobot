"""Forecast sanity bound: EBITDA margin capped at 5-year peak + 1 sd, with an explicit flagged override."""
import os
import statistics
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules.financial_data_processor import apply_margin_cap, calculate_growth_and_forecasts  # noqa: E402

CFG = {"revenue_base_year": "2026A", "revenue_growth_assumptions": {"2027E": 0.05, "2028E": 0.06, "2029E": 0.04},
       "margin_improvement": {"Contribution Margin": 0.01, "EBITDA Margin": 0.01}, "sga_margin_change": -0.005}


def _table(rev, cp, sga, ebitda, years=("2022A", "2023A", "2024A", "2025A", "2026A")):
    """Analysis table in absolute rupees with every row the forecaster needs."""
    pct = lambda a, b: [f"{x / y * 100:.1f}%" for x, y in zip(a, b)]
    growth = ["N/A"] + [f"{(rev[i] / rev[i - 1] - 1) * 100:.1f}%" for i in range(1, len(rev))]
    cost = [r - c for r, c in zip(rev, cp)]
    rows = {"Revenue": rev, "Cost of Operations": cost, "SG&A": sga, "Contribution Profit": cp,
            "Contribution Margin": pct(cp, rev), "EBITDA": ebitda, "EBITDA Margin": pct(ebitda, rev),
            "SG&A Margin": pct(sga, rev), "Revenue Growth": growth, "EPS": [3.0] * len(rev), "PE Ratio": [20.0] * len(rev)}
    df = pd.DataFrame({"metrics": list(rows), **{y: [rows[k][i] for k in rows] for i, y in enumerate(years)}})
    return df.astype("object")


def mahamaya():
    # Mahamaya Steel, FY22-FY26, rupees (EBITDA margin 3.1, 2.1, 2.1, 2.3, 2.7 %)
    rev = [4953247000.0, 6497718000.0, 7838288000.0, 8017643000.0, 8828483000.0]
    cp = [905371000.0, 1129433000.0, 1366927000.0, 1504027000.0, 1838132000.0]
    eb = [154402000.0, 138107000.0, 163087000.0, 187164000.0, 236553000.0]
    return _table(rev, cp, [c - e for c, e in zip(cp, eb)], eb)


def welspun():
    rev = [93114.7e6, 80937.6e6, 96792.4e6, 105450.9e6, 93991.1e6]
    cp = [42174.0e6, 35797.3e6, 45693.6e6, 48695.5e6, 43459.4e6]
    eb = [13587.2e6, 7525.4e6, 13689.1e6, 12985.0e6, 7932.1e6]
    return _table(rev, cp, [c - e for c, e in zip(cp, eb)], eb)


def margin(df, col):
    g = lambda m: float(df.loc[df["metrics"] == m, col].iloc[0])
    return g("EBITDA") / g("Revenue") * 100


def test_uncapped_path_would_reach_7_2_percent_for_mahamaya():
    raw = calculate_growth_and_forecasts(mahamaya(), {**CFG}, income_df=None)     # cap would bind, so read the flag's uncapped value
    caps = raw.attrs["margin_cap"]
    assert caps["years"]["2029E"]["uncapped_margin"] == pytest.approx(7.2, abs=0.1)
    assert caps["years"]["2027E"]["uncapped_margin"] == pytest.approx(4.2, abs=0.1)


def test_cap_equals_peak_plus_one_sample_sd():
    df = mahamaya()
    margins = [237 / 8828 * 100 for _ in range(1)]  # placeholder to keep the formula visible below
    ms = [float(df.loc[df["metrics"] == "EBITDA", y].iloc[0]) / float(df.loc[df["metrics"] == "Revenue", y].iloc[0]) * 100
          for y in ("2022A", "2023A", "2024A", "2025A", "2026A")]
    out = calculate_growth_and_forecasts(df, {**CFG})
    c = out.attrs["margin_cap"]
    assert c["peak"] == pytest.approx(max(ms)) and c["peak"] == pytest.approx(3.1, abs=0.05)
    assert c["std"] == pytest.approx(statistics.stdev(ms))
    assert c["cap"] == pytest.approx(max(ms) + statistics.stdev(ms))


def test_forecast_margins_are_bounded_and_the_table_reconciles():
    out = calculate_growth_and_forecasts(mahamaya(), {**CFG})
    c = out.attrs["margin_cap"]
    assert c["binding"] and not c["lifted"] and c["trigger"] is None
    for y in ("2027E", "2028E", "2029E"):
        assert margin(out, y) == pytest.approx(c["cap"], abs=1e-6)
        rev = float(out.loc[out["metrics"] == "Revenue", y].iloc[0])
        cp = float(out.loc[out["metrics"] == "Contribution Profit", y].iloc[0])
        sga = float(out.loc[out["metrics"] == "SG&A", y].iloc[0])
        eb = float(out.loc[out["metrics"] == "EBITDA", y].iloc[0])
        assert cp - sga == pytest.approx(eb, rel=1e-6)             # the cost bucket absorbed the difference
        assert out.loc[out["metrics"] == "EBITDA Margin", y].iloc[0] == f"{c['cap']:.1f}%"
    assert "capped at 3.6%" in c["message"] and "historical peak 3.1%" in c["message"] and "no fundamental trigger" in c["message"]


def test_material_margin_forecast_is_kept_and_other_costs_absorb():
    out = calculate_growth_and_forecasts(mahamaya(), {**CFG})
    assert out.loc[out["metrics"] == "Contribution Margin", "2029E"].iloc[0] == "23.8%"     # unchanged by the cap


def test_trigger_lifts_the_cap_and_is_recorded():
    out = calculate_growth_and_forecasts(mahamaya(), {**CFG, "margin_trigger": "new rolling mill commissioned in FY28"})
    c = out.attrs["margin_cap"]
    assert c["lifted"] and c["binding"] and c["trigger"] == "new rolling mill commissioned in FY28"
    assert margin(out, "2029E") == pytest.approx(7.2, abs=0.1)                  # the path is not changed
    assert "lifted by a stated trigger" in c["message"] and "rolling mill" in c["message"]


def test_cap_does_not_bind_for_welspun_like_margins():
    out = calculate_growth_and_forecasts(welspun(), {**CFG})
    c = out.attrs["margin_cap"]
    assert c["peak"] == pytest.approx(14.6, abs=0.05) and c["cap"] > 17
    assert not c["binding"] and c["message"] is None
    assert margin(out, "2029E") == pytest.approx(12.9, abs=0.1)


def test_not_enough_history_means_no_cap():
    short = calculate_growth_and_forecasts(_table([100e6, 110e6], [20e6, 22e6], [15e6, 16e6], [5e6, 6e6], ("2025A", "2026A")),
                                           {**CFG, "revenue_base_year": "2026A"})
    assert short.attrs.get("margin_cap") is None
    assert apply_margin_cap(short, ["2025A", "2026A"], ["2027E"]) is None


def test_eps_bridge_uses_the_capped_ebitda():
    income = pd.DataFrame([{"year": 2026, "date": pd.Timestamp("2026-03-31"), "revenue": 8828.483e6, "ebitda": 236.553e6,
                            "depreciationAndAmortization": 88.1e6, "interestExpense": 50.2e6, "otherIncome": 15.8e6,
                            "incomeBeforeTax": 114.1e6, "incomeTaxExpense": 18.1e6, "netIncome": 95.988e6,
                            "netIncomeIncludingMinority": 96.0e6, "eps": 5.84, "weightedAverageShsOutDil": 16.4e6}])
    capped = calculate_growth_and_forecasts(mahamaya(), {**CFG}, income_df=income, current_price=1500.0)
    free = calculate_growth_and_forecasts(mahamaya(), {**CFG, "margin_trigger": "x"}, income_df=income, current_price=1500.0)
    assert capped.attrs["forecast_bridge"].loc["2029E", "ebitda"] == pytest.approx(
        capped.attrs["margin_cap"]["cap"] / 100 * capped.attrs["forecast_bridge"].loc["2029E", "revenue"])
    eps_c = float(capped.loc[capped["metrics"] == "EPS", "2029E"].iloc[0])
    eps_f = float(free.loc[free["metrics"] == "EPS", "2029E"].iloc[0])
    assert eps_c < 0.6 * eps_f                                                    # the cap cuts the EPS path sharply
    pe_c = float(capped.loc[capped["metrics"].astype(str).str.startswith("PE Ratio"), "2029E"].iloc[0])
    assert pe_c == pytest.approx(1500.0 / eps_c, abs=0.1)
