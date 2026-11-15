#!/usr/bin/env python
# coding: utf-8
"""
Panel headlines and the one-sentence thesis, built from computed facts and verdicts (templates, no LLM).

Every headline states a conclusion in 8 words or fewer. ``HEADLINE_RULES`` documents the logic so a reviewer can read it.
"""

from __future__ import annotations

from typing import Optional

WORDS = {1: "One", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _v(chips: dict, key: str) -> Optional[str]:
    c = chips.get(key)
    return c["verdict"] if c else None


def profit_engine(chips: dict, facts: dict) -> str:
    cm, sga = _v(chips, "contribution_margin"), _v(chips, "sga_pct")
    if cm in ("good", "neutral") and sga == "concern":
        return "Prices held. Costs broke the margin."
    if cm == "concern" and sga == "concern":
        return "Margins squeezed from both sides."
    if cm == "concern":
        return "Pricing weakened; overheads held."
    if sga == "good" and cm == "good":
        return "Pricing strong, overheads lean."
    return "Margins steady; profit follows sales."


def cyclicality(chips: dict, facts: dict) -> str:
    downs, n = facts.get("downturn_years", 0), facts.get("growth_observations", 0)
    same = facts.get("downturns_hit_margin", False)
    if downs == 0:
        return "No downturn in the window; steady growth."
    if downs == 1:
        return f"One downturn in {NUM_WORDS.get(n, n)} years."
    return f"{WORDS.get(downs, str(downs) + 'th')} downturn in {NUM_WORDS.get(n, n)} years" + (", same pattern." if same else ".")


def strength(chips: dict, facts: dict) -> str:
    de = facts.get("debt_equity_trend")      # "falling" | "rising" | "steady"
    flag, conv = facts.get("cash_cyclical_flag"), facts.get("conversion_latest_pct")
    debt = {"falling": "Debt falling", "rising": "Debt rising", "steady": "Debt steady"}.get(de, "Debt unchanged")
    if conv is None:
        cash = "cash data limited"
    elif flag and conv >= 100:
        cash = "cash strong, partly one-off"
    elif conv < 60:
        cash = "cash conversion weak"
    else:
        cash = "cash solid"
    cov = _v(chips, "interest_coverage")
    if cov == "concern" and "one-off" not in cash:
        cash += ", cover thin"
    return f"{debt}; {cash}."


def valuation(chips: dict, facts: dict) -> str:
    hist, peer = facts.get("ev_vs_history"), facts.get("ev_vs_peers")       # "above" | "below" | "within"
    steady = facts.get("steadiest_peer_below_subject")
    h = {"above": "above its history", "below": "below its history", "within": "near its history"}.get(hist)
    p = {"above": "above peers", "below": "below peers", "within": "in line with peers"}.get(peer)
    if steady and peer == "above":
        p = "above its steadiest peer"
    if h and p:
        return f"Priced {h} and {p}."
    return f"Priced {h or p or 'without clear benchmarks'}."


def price_behaviour(tests: dict, facts: dict) -> str:
    trend, mom = tests.get("trend"), tests.get("momentum")
    stretch = tests.get("stretch")
    if trend == "good" and mom in ("neutral", "concern"):
        return "Strong trend, losing force."
    if trend == "good" and stretch == "concern":
        return "Strong trend, but stretched."
    if trend == "good":
        return "Uptrend with momentum behind it."
    if trend == "concern" and mom == "concern":
        return "Downtrend with no momentum."
    if trend == "concern":
        return "Downtrend, momentum stabilising."
    return "No clear trend; wait for confirmation."


def watch(n: int, facts: dict) -> str:
    if facts.get("recovery_narrative"):
        return f"{NUM_WORDS.get(n, n).capitalize()} signs the recovery is real."
    return f"{NUM_WORDS.get(n, n).capitalize()} numbers decide the next year."


def summary(lights: dict) -> str:
    biz = "Sound business" if lights.get("business") != "concern" else "Fragile business"
    prof = {"concern": "profit squeezed", "neutral": "profit under pressure", "good": "profit growing"}.get(lights.get("profit"), "profit unclear")
    price = {"concern": "rich price", "neutral": "fair price", "good": "cheap price"}.get(lights.get("price"), "price unclear")
    return f"{biz}, {prof}, {price}."


def thesis(chips: dict, facts: dict, lights: dict) -> str:
    cm, sga = _v(chips, "contribution_margin"), _v(chips, "sga_pct")
    part1 = "A sound product business" if cm in ("good", "neutral") else "A business with thin pricing power"
    part2 = {"concern": ", hurt by fixed costs", "good": ", running lean"}.get(sga, ", with overheads in a normal range")
    recovering = facts.get("recovery_narrative")
    price = _v(chips, "fair_value") or lights.get("price")
    if price == "concern":
        part3 = ", priced for a recovery that has only started" if recovering else ", priced richly against its forecasts"
    elif price == "good":
        part3 = ", priced below its fair-value range"
    else:
        part3 = ", priced in line with its fair-value range"
    return f"{part1}{part2}{part3}."


HEADLINE_RULES = {
    "profit_engine": "contribution margin verdict x SG&A verdict",
    "cyclicality": "count of negative revenue-growth years in the window and whether EBITDA margin fell in each",
    "strength": "debt to equity trend, cash conversion level and the cyclicality flag",
    "valuation": "EV/EBITDA against own history and peers, and the steadiest peer",
    "price": "trend and momentum tests, then stretch",
    "watch": "number of signposts and whether TTM earnings are recovering",
}
