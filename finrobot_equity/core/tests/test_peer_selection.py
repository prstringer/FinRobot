"""Peer selection: sub-sector tag + market-cap band, insufficient peers, explicit peers, margin profile check."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules import peer_selection as ps  # noqa: E402


class Stub:
    """Minimal FinEdge client: market caps in crore, peer lists per group, profile sector."""

    def __init__(self, caps, pool, sector="Iron & Steel Products"):
        self.caps, self.pool, self.sector = caps, pool, sector

    def quote(self, s):
        return {"market_cap": self.caps.get(s, 0)}

    def peers(self, s, group="sector"):
        return list(self.pool.get(group, self.pool.get("sector", [])))

    def profile(self, s):
        return {"sector": self.sector}


def stub(caps, pool=None):
    return Stub(caps, {"sub_industry": pool or [], "sector": pool or []})


def test_subsector_map_loads_and_separates_long_from_pipes():
    assert ps.subsector_of("MAHASTEEL") == "steel_long_sponge"
    assert ps.subsector_of("APLAPOLLO") == "steel_pipes"
    assert ps.subsector_of("ITC") is None
    assert "Long Products" in ps.subsector_label("steel_long_sponge") and "Pipes" in ps.subsector_label("steel_pipes")


def test_band_edges_are_inclusive():
    caps = {"MAHASTEEL": 1000.0, "GALLANTT": 200.0, "GPIL": 5000.0, "SHYAMMETL": 199.9, "KAMDHENU": 5000.1}
    r = ps.select_peers("MAHASTEEL", stub(caps, list(caps)), min_peers=1)
    assert set(r["peers"]) == {"GALLANTT", "GPIL"}                    # exactly 0.2x and 5.0x kept
    why = {x["ticker"]: x["reason"] for x in r["rejected"]}
    assert "outside" in why["SHYAMMETL"] and "outside" in why["KAMDHENU"]


def test_pipe_makers_are_rejected_for_a_long_products_subject():
    caps = {"MAHASTEEL": 2498.0, "APLAPOLLO": 62473.0, "JINDALSAW": 16720.0, "GALLANTT": 3000.0, "GPIL": 2500.0, "KAMDHENU": 4000.0}
    r = ps.select_peers("MAHASTEEL", stub(caps, ["APLAPOLLO", "JINDALSAW", "GALLANTT"]))
    assert r["peers"][:3] == ["GPIL", "GALLANTT", "KAMDHENU"] or set(r["peers"]) == {"GPIL", "GALLANTT", "KAMDHENU"}
    why = {x["ticker"]: x["reason"] for x in r["rejected"]}
    assert "different sub-sector" in why["APLAPOLLO"] and "Pipes" in why["APLAPOLLO"]
    assert not r["insufficient"]


def test_ordering_is_by_closeness_in_log_market_cap():
    caps = {"MAHASTEEL": 1000.0, "GALLANTT": 4000.0, "GPIL": 1100.0, "SHYAMMETL": 400.0, "KAMDHENU": 1900.0}
    r = ps.select_peers("MAHASTEEL", stub(caps, list(caps)), max_peers=3)
    assert r["peers"] == ["GPIL", "KAMDHENU", "SHYAMMETL"]          # ratios 1.1, 1.9, 0.4 -> nearest first
    assert any(x["ticker"] == "GALLANTT" and "closest" in x["reason"] for x in r["rejected"])


def test_insufficient_peers_is_flagged_not_widened():
    caps = {"MAHASTEEL": 2498.0, "SHYAMMETL": 36838.0, "GPIL": 14824.0, "GALLANTT": 12618.0, "APLAPOLLO": 62473.0}
    r = ps.select_peers("MAHASTEEL", stub(caps, list(caps)))
    assert r["peers"] == [] and r["insufficient"] is True
    assert any("not reliable" in w for w in r["warnings"])
    reasons = {x["ticker"]: x["reason"] for x in r["rejected"]}
    assert "outside" in reasons["SHYAMMETL"] and "outside" in reasons["GPIL"] and "different sub-sector" in reasons["APLAPOLLO"]


def test_explicit_peers_are_kept_but_outliers_are_warned():
    caps = {"WELSPUNLIV": 23524.0, "KPRMILL": 52000.0, "TRIDENT": 12000.0, "ICIL": 3000.0, "GARFIBRES": 1000.0}
    r = ps.select_peers("WELSPUNLIV", stub(caps), explicit=["KPRMILL", "TRIDENT", "ICIL", "GARFIBRES", "WELSPUNLIV"])
    assert r["peers"] == ["KPRMILL", "TRIDENT", "ICIL", "GARFIBRES"] and r["explicit"] is True
    assert any("ICIL" in w and "outside" in w for w in r["warnings"])
    assert any("GARFIBRES" in w for w in r["warnings"])
    assert not any("KPRMILL" in w for w in r["warnings"])             # 2.2x is inside the band
    assert r["insufficient"] is False


def test_untagged_subject_falls_back_to_the_finedge_list_and_says_so():
    caps = {"ITC": 100000.0, "HUL": 150000.0, "NESTLEIND": 220000.0, "TINY": 100.0}
    r = ps.select_peers("ITC", stub(caps, ["HUL", "NESTLEIND", "TINY"]), min_peers=2)
    assert r["subsector"] is None and "no sub-sector tag" in r["basis"]
    assert set(r["peers"]) == {"HUL", "NESTLEIND"}
    assert any("TINY" == x["ticker"] for x in r["rejected"])
    assert any("no sub-sector tag" in w for w in r["warnings"])


def test_missing_market_cap_is_rejected_with_a_reason():
    caps = {"MAHASTEEL": 1000.0, "GPIL": 1500.0}
    r = ps.select_peers("MAHASTEEL", stub(caps, ["GPIL", "GALLANTT"]), min_peers=1)
    assert any(x["ticker"] == "GALLANTT" and x["reason"] == "market cap unavailable" for x in r["rejected"])


def test_margin_profile_check_flags_a_factor_or_a_gap():
    mh = ps.margin_profile_check(2.7, {"A": 12.0, "B": 13.0, "C": 12.6})
    assert mh["mismatch"] and mh["peer_median_margin_pct"] == 12.6 and "differ" in mh["warning"]
    assert ps.margin_profile_check(14.0, {"A": 12.0, "B": 16.0})["mismatch"] is False
    assert ps.margin_profile_check(3.0, {"A": 9.5, "B": 10.0})["mismatch"]         # 3.2x apart and 6.5 points
    assert ps.margin_profile_check(10.0, {"A": 16.0, "B": 17.0})["mismatch"]        # 6.5 points apart, under 3x
    assert ps.margin_profile_check(-2.0, {"A": 10.0})["mismatch"]                    # opposite signs
    assert ps.margin_profile_check(None, {"A": 10.0})["mismatch"] is False


def test_summary_lines_are_readable():
    caps = {"MAHASTEEL": 1000.0, "GPIL": 50000.0}
    r = ps.select_peers("MAHASTEEL", stub(caps, ["GPIL"]))
    text = "\n".join(ps.summary_lines(r, ps.margin_profile_check(2.7, {"X": 12.0})))
    assert "Kept (0)" in text and "Rejected GPIL" in text and "WARNING" in text
