#!/usr/bin/env python
# coding: utf-8
"""
LLM classification of news items: sentiment, impact and event type.

The keyword rules in the news integrator and catalyst analyzer call most headlines neutral
("shares rally 20%" and "shares fall amid weakness" both come out neutral) and treat every
routine exchange filing as a high-impact "regulatory" catalyst. One small, cheap model call
over the whole list fixes both. The result is stored on each item under ``llm_sentiment``,
``llm_impact`` and ``llm_event_type``; the keyword rules remain the fallback when this step
is unavailable or fails.
"""

from __future__ import annotations

import json
import re

from modules.llm_providers import LLMSettings, generate_text

SENTIMENTS = ("positive", "negative", "neutral")
IMPACTS = ("high", "medium", "low")
EVENT_TYPES = ("earnings", "regulatory", "acquisition", "management", "product_launch", "market", "other")
MAX_ITEMS = 40

SYSTEM_PROMPT = (
    "You are an equity research analyst. For each numbered news item about the company, judge it "
    "from a shareholder's point of view. Reply with JSON only (no markdown, no commentary) in this form: "
    '{"items": [{"n": 1, "sentiment": "positive|negative|neutral", "impact": "high|medium|low", '
    '"event_type": "earnings|regulatory|acquisition|management|product_launch|market|other"}]}. '
    "Rules: routine exchange filings (ESOP allotments, director or committee changes, meeting schedules, "
    "procedural disclosures, certificates) are impact low and event_type other. Share-price moves and "
    "market commentary are event_type market. Only material results, deals, regulatory actions, "
    "guidance or management changes are medium or high impact. Include every item."
)


def _format_items(news: list[dict]) -> str:
    lines = []
    for i, a in enumerate(news[:MAX_ITEMS], 1):
        lines.append(f"{i}. [{a.get('site', '')}, {(a.get('publishedDate') or '')[:10]}] {a.get('title', '')}")
    return "\n".join(lines)


def parse_response(text: str, count: int) -> dict[int, dict]:
    """Extract {item number -> validated fields} from the model's JSON reply ({} if unusable)."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        items = json.loads(match.group(0)).get("items", [])
    except (ValueError, AttributeError):
        return {}
    out = {}
    for it in items:
        try:
            n = int(it.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        sentiment = str(it.get("sentiment", "")).lower()
        impact = str(it.get("impact", "")).lower()
        event = str(it.get("event_type", "")).lower()
        if 1 <= n <= count and sentiment in SENTIMENTS and impact in IMPACTS and event in EVENT_TYPES:
            out[n] = {"llm_sentiment": sentiment, "llm_impact": impact, "llm_event_type": event}
    return out


def annotate_news_with_llm(news: list[dict], settings: LLMSettings | None, company_name: str = "",
                           usage_sink: list | None = None) -> int:
    """Add llm_sentiment / llm_impact / llm_event_type to the news dicts in place.

    Returns how many items were annotated (0 if there is no LLM, no news, or the call failed).
    """
    if not news or settings is None or not settings.api_key:
        return 0
    batch = news[:MAX_ITEMS]
    prompt = f"Company: {company_name}\n\nNews items:\n{_format_items(batch)}"
    try:
        reply = generate_text(settings, SYSTEM_PROMPT, prompt, max_tokens=3000, usage_sink=usage_sink,
                              section="news_analysis")
    except Exception as e:
        print(f"Warning: AI news classification failed, using keyword rules instead: {e}")
        return 0
    parsed = parse_response(reply, len(batch))
    for n, fields in parsed.items():
        batch[n - 1].update(fields)
    if not parsed:
        print("Warning: AI news classification returned nothing usable, using keyword rules instead")
    return len(parsed)
