#!/usr/bin/env python
# coding: utf-8
"""
Sentiment gate: quantitative valuation outranks headline tone.

If the implied price is below 70% of the current price (a downside of more than 30%), the report's overall sentiment flag
is forced to CONTRADICTORY/BEARISH, whatever the news tone says. A positive quarterly headline cannot sit next to a large
valuation downside without that conflict being shown.

Implied price = median of the fair-value cases in the report context (the model EBITDA at 12x to 18x, net of net debt,
per share). The same function serves every output format, so they cannot disagree.
"""

from __future__ import annotations

import statistics
from typing import Optional

THRESHOLD = 0.70
FLAG = "CONTRADICTORY/BEARISH"
FLAG_KEY = "contradictory/bearish"      # lower-case form stored in the news summary (renderers upper-case it for display)


def implied_price(ctx: dict) -> Optional[float]:
    grid = ((ctx or {}).get("fair_value") or {}).get("grid") or []
    prices = [g["implied_price"] for g in grid if g.get("implied_price") is not None]
    return statistics.median(prices) if prices else None


def gate_status(ctx: dict, threshold: float = THRESHOLD) -> dict:
    price = ((ctx or {}).get("market") or {}).get("price")
    implied = implied_price(ctx)
    if not price or implied is None:
        return {"available": False, "triggered": False}
    return {"available": True, "triggered": implied < price * threshold, "implied_price": implied, "current_price": price,
            "threshold_price": price * threshold, "ratio": implied / price, "downside_pct": (implied / price - 1) * 100,
            "threshold": threshold}


def gate_note(status: dict) -> str:
    return (f"Implied price Rs {status['implied_price']:,.0f} is {abs(status['downside_pct']):.0f}% below the current price of "
            f"Rs {status['current_price']:,.0f}, which is under the {status['threshold']:.0%} gate (Rs {status['threshold_price']:,.0f}). "
            "Sentiment is forced to CONTRADICTORY/BEARISH regardless of headline tone.")


def apply_gate(enhanced_news: Optional[dict], ctx: dict) -> Optional[dict]:
    """Override ``enhanced_news['sentiment_summary']['overall']`` in place when the gate triggers. Returns the gate status."""
    status = gate_status(ctx)
    if not enhanced_news or not status["available"]:
        return status
    ss = enhanced_news.setdefault("sentiment_summary", {}) if isinstance(enhanced_news.get("sentiment_summary", {}), dict) else {}
    if "news_tone" not in ss:
        ss["news_tone"] = ss.get("overall", "neutral")        # keep the headline tone for transparency
    if status["triggered"]:
        ss["overall"] = FLAG_KEY
        ss["gate_note"] = gate_note(status)
    else:
        ss["overall"] = ss["news_tone"]
        ss.pop("gate_note", None)
    ss["gate"] = status
    enhanced_news["sentiment_summary"] = ss
    return status


def gate_fact_block(status: dict, news_tone: Optional[str]) -> Optional[str]:
    """Fact line for the news-summary prompt, so the narrative cannot call the news positive without the conflict."""
    if not status.get("triggered"):
        return None
    return ("SENTIMENT GATE (computed): headline tone is " + (news_tone or "n/a") + ", but " + gate_note(status)
            + " Say plainly that good headlines conflict with the valuation downside.")


def display_label(overall: str) -> str:
    return FLAG if str(overall).lower() == FLAG_KEY else str(overall).upper()
