"""Google News headlines for Indian stocks: parsing, relevance filtering and de-duplication."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules import google_news as gn

FEED = b"""<?xml version="1.0"?><rss><channel>
<item><title>ITC shares slide 4% after block deal - BusinessLine</title><source>BusinessLine</source>
  <pubDate>Fri, 16 Oct 2026 08:00:00 GMT</pubDate><link>http://a/1</link></item>
<item><title>ITC Share Price - Upstox</title><source>Upstox</source>
  <pubDate>Sat, 14 Nov 2026 08:00:00 GMT</pubDate><link>http://a/2</link></item>
<item><title>ITC shares slide 4% after block deal - Mint</title><source>Mint</source>
  <pubDate>Fri, 16 Oct 2026 09:00:00 GMT</pubDate><link>http://a/3</link></item>
<item><title>Nifty ends higher as banks rally - Reuters</title><source>Reuters</source>
  <pubDate>Mon, 03 Nov 2026 08:00:00 GMT</pubDate><link>http://a/4</link></item>
<item><title>ITC Q2 profit drops 20%, brokerages flag cigarette pressure - Mint</title><source>Mint</source>
  <pubDate>Tue, 03 Nov 2026 10:00:00 GMT</pubDate><link>http://a/5</link></item>
</channel></rss>"""


def test_parse_feed_filters_dedupes_and_sorts():
    items = gn.parse_feed(FEED, "ITC Limited", "ITC")
    titles = [i["title"] for i in items]
    assert titles == ["ITC Q2 profit drops 20%, brokerages flag cigarette pressure",
                      "ITC shares slide 4% after block deal"]  # quote page, unrelated and duplicate dropped, newest first
    first = items[0]
    assert first["symbol"] == "ITC" and first["site"] == "Mint" and first["publishedDate"].startswith("2026-11-03")


def test_relevance_uses_company_words_not_just_ticker():
    # The ticker HDFCBANK never appears in headlines, the company name does
    assert gn.is_relevant("HDFC Bank profit rises 11%", "HDFC Bank Limited", "HDFCBANK")
    assert not gn.is_relevant("Axis Bank profit rises 11%", "HDFC Bank Limited", "HDFCBANK")
    # a generic first word must not match every group company
    assert not gn.is_relevant("Tata Motors sales fall", "Tata Consultancy Services Ltd", "TCS")
    assert gn.is_relevant("Tata Consultancy Services wins deal", "Tata Consultancy Services Ltd", "TCS")


def test_bad_xml_returns_empty():
    assert gn.parse_feed(b"not xml", "ITC Limited", "ITC") == []


def test_network_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(gn.requests, "get", boom)
    assert gn.get_google_news("ITC Limited", "ITC") == []


# ---- news pipeline regressions ----------------------------------------------

def _filing(days_ago=2):
    from datetime import datetime, timedelta
    d = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
    return {"symbol": "HDFCBANK", "title": "Change in Director(s): HDFC Bank Limited has informed the Exchange regarding Change in Directors",
            "publishedDate": d, "text": "HDFC Bank Limited has informed the Exchange", "site": "NSE/BSE filing"}


def test_filings_scoped_to_the_symbol_are_not_dropped_by_the_relevance_filter():
    """HDFCBANK filings say 'HDFC Bank Limited' (never the ticker) and used to be filtered out."""
    from modules.news_integrator import build_enhanced_news
    out = build_enhanced_news("HDFCBANK", [_filing()], days_back=30, company_name="HDFC Bank Limited")
    assert len(out["articles"]) == 1


def test_enhanced_news_exposes_sentiment_summary_the_templates_read():
    from datetime import datetime
    from modules.news_integrator import build_enhanced_news
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    news = [{"symbol": "ITC", "title": "ITC profit jumps, brokerages upgrade to buy", "text": "ITC strong growth beat", "publishedDate": now, "site": "Mint"},
            {"symbol": "ITC", "title": "ITC shares fall after profit warning", "text": "ITC weak outlook decline", "publishedDate": now, "site": "Mint"}]
    out = build_enhanced_news("ITC", news, days_back=30, company_name="ITC Limited")
    s = out["sentiment_summary"]
    assert s["overall"] in {"positive", "negative", "neutral"}
    assert s["positive"] + s["negative"] + s["neutral"] == len(out["articles"])


def test_empty_retail_sentiment_block_is_hidden():
    from modules.html_template_professional import format_retail_sentiment_html_professional
    from modules.html_renderer import format_retail_sentiment_html
    from modules.retail_sentiment_client import has_coverage
    empty = {"coverage": "0/3", "coverage_ratio": 0.0, "average_buzz": None, "sources": [{"key": "reddit", "has_data": False}]}
    assert not has_coverage(empty) and not has_coverage(None) and not has_coverage({})
    assert format_retail_sentiment_html_professional(empty) == ""
    assert format_retail_sentiment_html(empty) == ""
    assert has_coverage({"coverage_ratio": 0.33})


def test_news_prompt_lists_headlines_before_filings():
    from modules import text_generator_agents as tga
    filing = dict(_filing(), title="Filing A", publishedDate="2026-11-13 10:00:00")
    headline = {"symbol": "ITC", "title": "Headline B", "publishedDate": "2026-11-01 10:00:00", "text": "x", "site": "Mint"}
    p = tga._prepare_user_prompt({"company_news": [filing, headline]}, "news_summary", "Acme", "ACME")
    assert p.index("Headline B") < p.index("Filing A")


def test_quote_pages_are_not_news():
    for t in ["HDFC Bank Limited (HDB) Stock Price, News, Quote & History",
              "HDFC Bank Share Price - Live NSE: HDFCBANK Stock Price & Chart",
              "ITC Share Price", "ITC Ltd Share Price Today"]:
        assert gn._QUOTE_PAGE.search(t), t
    for t in ["ITC shares slide 4% to near 52-week low after block deal",
              "HDFC Bank share price target raised by Jefferies"]:
        assert not gn._QUOTE_PAGE.search(t), t
