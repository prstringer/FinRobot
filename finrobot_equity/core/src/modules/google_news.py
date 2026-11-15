#!/usr/bin/env python
# coding: utf-8
"""
Recent news headlines for a company from the Google News RSS feed.

Free and key-less, used to give Indian (FinEdge) reports real news on top of the exchange
filings FinEdge provides. Items use the same shape as the other news sources
(symbol, title, publishedDate, text, site, url) so the news summary, news integrator and
catalyst analyzer can consume them unchanged.

Any network or parsing problem returns an empty list: news is an enhancement and must never
break a report run.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import requests

FEED_URL = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
_GENERIC_WORDS = {"limited", "ltd", "ltd.", "inc", "inc.", "corp", "corp.", "co", "co.", "the", "and",
                  "of", "group", "holdings", "plc", "india", "industries", "company"}
# Headlines that are only a quote-page title, e.g. "ITC Share Price" or "HDFC Bank Stock Price Today"
_QUOTE_PAGE = re.compile(
    r"^[\w\s&.\-']+?\s(share|stock) price(\s+(today|live|chart|history|news|nse|bse))*\s*$"
    r"|(share|stock) price\s*[,:\-\u2013]\s*(news|live|today|chart|quote)"
    r"|\blive (nse|bse)\b|quote\s*&\s*history|price\s*&\s*chart",
    re.I)


def _short_name(company_name: str) -> str:
    """'ITC Limited' -> 'ITC', 'Hitachi Energy India Ltd' -> 'Hitachi Energy India'."""
    words = [w for w in (company_name or "").split() if w.lower().strip(".,") not in {"limited", "ltd", "inc", "corp"}]
    return " ".join(words) or (company_name or "")


def _distinctive_tokens(company_name: str, limit: int = 2) -> list[str]:
    """Lower-case words that identify the company (first ``limit`` non-generic words)."""
    words = [re.sub(r"[^\w&]", "", w).lower() for w in (company_name or "").split()]
    return [w for w in words if w and w not in _GENERIC_WORDS and len(w) > 1][:limit]


def is_relevant(title: str, company_name: str, ticker: str) -> bool:
    """True if the headline names the company (all of its first distinctive words) or its ticker."""
    t = (title or "").lower()
    tokens = _distinctive_tokens(company_name)
    if tokens and all(tok in t for tok in tokens):
        return True
    return bool(ticker) and re.search(rf"\b{re.escape(ticker.lower())}\b", t) is not None


def is_english(title: str, max_non_latin: float = 0.10) -> bool:
    """False when more than 10% of the letters are outside the Latin scripts (e.g. a Hindi headline from an Indian feed).

    The reports are written in English and their PDFs use a Latin font, so such headlines can be neither summarised
    reliably nor printed; they are dropped at the source.
    """
    letters = [c for c in (title or "") if c.isalpha()]
    if not letters:
        return True
    return sum(1 for c in letters if ord(c) > 0x024F) / len(letters) <= max_non_latin


_TEXT_KEYS = ("title", "description", "text", "headline")


def strip_non_english(obj):
    """Remove non-English headlines from already-saved news, catalyst and enhanced-news data (older outputs).

    Lists of articles or catalysts lose any entry whose title/description is not English; multi-line summaries lose the
    offending line and the "Impact:" line that belongs to it. Everything else is returned unchanged.
    """
    if isinstance(obj, list):
        out = []
        for x in obj:
            if isinstance(x, dict) and any(isinstance(x.get(k), str) and not is_english(x[k]) for k in _TEXT_KEYS):
                continue
            out.append(strip_non_english(x))
        return out
    if isinstance(obj, dict):
        return {k: strip_non_english(v) for k, v in obj.items()}
    if isinstance(obj, str) and "\n" in obj and not is_english(obj, 0.0):
        lines, keep, skip_impact = obj.split("\n"), [], False
        for ln in lines:
            if not is_english(ln):
                skip_impact = True
                continue
            if skip_impact and ln.strip().startswith(("- Impact", "Impact")):
                skip_impact = False
                continue
            skip_impact = False
            keep.append(ln)
        return "\n".join(keep)
    return obj


def parse_feed(xml_bytes: bytes, company_name: str, ticker: str) -> list[dict]:
    """Turn a Google News RSS document into news items (newest first, de-duplicated)."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    items, seen = [], set()
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        publisher = (item.findtext("source") or "").strip()
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3].rstrip()
        if not title or not is_english(title) or _QUOTE_PAGE.search(title) or not is_relevant(title, company_name, ticker):
            continue
        key = re.sub(r"[^a-z0-9]", "", title.lower())[:50]
        if key in seen:
            continue
        seen.add(key)
        try:
            dt = parsedate_to_datetime(item.findtext("pubDate") or "")
            published = dt.astimezone(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            published = ""
        items.append({
            "symbol": ticker,
            "title": title,
            "publishedDate": published,
            "text": title,
            "site": publisher or "Google News",
            "url": (item.findtext("link") or "").strip(),
            "category": "news",
        })
    items.sort(key=lambda a: a["publishedDate"], reverse=True)
    return items


def get_google_news(company_name: str, ticker: str, days_back: int = 30, limit: int = 15) -> list[dict]:
    """Recent headlines about the company (empty list if the feed is unreachable)."""
    query = quote_plus(f"{_short_name(company_name)} share when:{max(1, int(days_back))}d")
    try:
        resp = requests.get(FEED_URL.format(query=query), timeout=15,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; equity-research-report)"})
        resp.raise_for_status()
    except Exception as e:
        print(f"Warning: could not fetch Google News headlines for {ticker}: {e}")
        return []
    return parse_feed(resp.content, company_name, ticker)[:limit]
