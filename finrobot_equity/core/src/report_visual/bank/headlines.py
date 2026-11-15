#!/usr/bin/env python
# coding: utf-8
"""
Bank panel headlines and thesis: templates over verdicts and computed facts. At most 8 words each, conclusion style.
Rules are documented in variants/bank.yaml (headline_rules). The price and watch headlines reuse the shared functions.
"""

from __future__ import annotations

from typing import Optional

from .. import headlines as shared

NUM_WORDS = shared.NUM_WORDS
ORD = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth"}


def _v(chips: dict, key: str) -> Optional[str]:
    c = chips.get(key)
    return c["verdict"] if c else None


def profit_engine(chips: dict, facts: dict) -> str:
    nim, ci, cc = _v(chips, "nim"), _v(chips, "cost_to_income"), _v(chips, "credit_cost")
    if cc == "concern":
        return "Bad-loan charges are eating the profit."
    if nim == "concern" and ci == "concern":
        return "Margin and costs both slipping."
    if nim == "concern":
        return "Margin narrowing; costs under control." if ci == "good" else "Margin narrowing; costs steady."
    if ci == "concern":
        return "Margin holds; costs are rising."
    if nim == "good" and ci == "good":
        return "Wider margin, tighter costs."
    return "Earnings steady; no margin pressure."


def cycle(chips: dict, facts: dict) -> str:
    downs, n = facts.get("downturn_years", 0), facts.get("growth_observations", 0)
    if downs >= 1:
        return f"{ORD.get(downs, str(downs) + 'th')} profit dip in {NUM_WORDS.get(n, n)} years."
    if facts.get("profit_growth_slowing"):
        return "Profit grew every year, but slowing."
    return "Profit grew every year in the window."


def strength(chips: dict, facts: dict) -> str:
    asset = {"good": "Loan quality clean", "neutral": "Loan quality steady", "concern": "Bad loans a concern"}.get(_v(chips, "gnpa"), "Loan quality unclear")
    cap = {"good": "capital strong", "neutral": "capital adequate", "concern": "capital thin"}.get(_v(chips, "cet1"), "capital unclear")
    return f"{asset}; {cap}."


def valuation(chips: dict, facts: dict) -> str:
    h = {"above": "above its history", "below": "below its history", "within": "near its history"}.get(facts.get("pb_vs_history"))
    p = {"above": "above peers", "below": "below peers", "within": "level with peers"}.get(facts.get("pb_vs_peers"))
    if h and p:
        return f"Priced {h} and {p}."
    return f"Priced {h or p or 'without clear benchmarks'}."


def thesis(chips: dict, facts: dict, lights: dict) -> str:
    biz = lights.get("business")
    part1 = "A well-capitalised lender" if biz in ("good", "neutral") and _v(chips, "cet1") != "concern" else "A lender under strain"
    nim = _v(chips, "nim")
    part2 = {"concern": ", with its lending margin narrowing", "good": ", with a widening lending margin"}.get(nim, ", with a steady lending margin")
    price = _v(chips, "fair_value") or lights.get("price")
    part3 = {"concern": ", priced above its fair-value range", "good": ", priced below its fair-value range"}.get(price, ", priced inside its fair-value range")
    return f"{part1}{part2}{part3}."


RULES = {"profit_engine": profit_engine, "cycle": cycle, "strength": strength, "valuation": valuation}


def headline(rule: str, chips: dict, facts: dict, lights: Optional[dict] = None, tests: Optional[dict] = None, n_signposts: int = 0) -> str:
    if rule == "summary":
        return shared.summary(lights or {})
    if rule == "price":
        return shared.price_behaviour(tests or {}, facts)
    if rule == "watch":
        return shared.watch(n_signposts, {"recovery_narrative": False})
    return RULES[rule](chips, facts)
