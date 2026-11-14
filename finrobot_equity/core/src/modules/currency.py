#!/usr/bin/env python
# coding: utf-8
"""
Report currency and number formatting.

The pipeline used to hard-code US dollars ("$", "USDb", values / 1e9). This
module holds the active currency so renderers and charts can format amounts
consistently. Default is USD, which keeps the FMP output unchanged.

INR amounts are shown in rupee crore (1 crore = 1e7), the usual unit for Indian
equity research: ₹89,913 Cr.

Usage:
    from modules import currency
    currency.set_currency("INR")
    currency.fmt_amount(899133000000)   # '₹89,913 Cr'
    currency.fmt_price(284.05)          # '₹284.05'
"""

from __future__ import annotations

import re

import pandas as pd

_CURRENCIES = {
    "USD": {
        "code": "USD", "symbol": "$",
        "large_div": 1e9, "large_suffix": "B", "large_label": "USDb",
        "mid_div": 1e6, "mid_suffix": "M",
        "chart_div": 1e9, "chart_label": "$B",
        "price_label": "USD",
    },
    "INR": {
        "code": "INR", "symbol": "₹",
        "large_div": 1e7, "large_suffix": " Cr", "large_label": "₹ Cr",
        "mid_div": 1e7, "mid_suffix": " Cr",
        "chart_div": 1e7, "chart_label": "₹ Cr",
        "price_label": "INR",
    },
}

_active = dict(_CURRENCIES["USD"])
_symbol_override: str | None = None


def set_currency(code: str) -> None:
    """Set the report currency ('USD' or 'INR')."""
    global _symbol_override
    code = (code or "USD").upper()
    if code not in _CURRENCIES:
        raise ValueError(f"Unsupported currency {code!r}; choose from {sorted(_CURRENCIES)}")
    _active.clear()
    _active.update(_CURRENCIES[code])
    _symbol_override = None


def set_symbol_override(text: str | None) -> None:
    """Use ``text`` instead of the currency sign (e.g. 'Rs ' for PDFs whose
    built-in fonts have no ₹ glyph). ``None`` restores the normal sign."""
    global _symbol_override
    _symbol_override = text


def get_currency() -> str:
    return _active["code"]


def symbol() -> str:
    return _symbol_override if _symbol_override is not None else _active["symbol"]


def price_label() -> str:
    """Label for per-share prices, e.g. 'USD' or 'INR'."""
    return _active["price_label"]


def large_label() -> str:
    """Label for the market-cap unit, e.g. 'USDb' or '₹ Cr'."""
    if get_currency() == "INR":
        return f"{symbol().strip()} Cr"
    return _active["large_label"]


def chart_scale() -> tuple[float, str]:
    """(divisor, axis label) for plotting absolute amounts."""
    if get_currency() == "INR":
        return _active["chart_div"], f"{symbol().strip()} Cr"
    return _active["chart_div"], _active["chart_label"]


def fmt_price(value, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    return f"{symbol()}{float(value):,.{decimals}f}"


def fmt_amount(value, decimals: int | None = None) -> str:
    """Format an absolute currency amount ($1.2B / $3.4M or ₹89,913 Cr)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    v = float(value)
    s = symbol()
    if get_currency() == "INR":
        crore = v / 1e7
        d = decimals if decimals is not None else (0 if abs(crore) >= 100 else 1)
        if abs(v) >= 1e5:
            return f"{s}{crore:,.{d}f} Cr"
        return f"{s}{v:,.0f}"
    if abs(v) >= 1e12:
        return f"{s}{v / 1e12:,.{decimals if decimals is not None else 2}f}T"
    if abs(v) >= 1e9:
        return f"{s}{v / 1e9:,.{decimals if decimals is not None else 1}f}B"
    if abs(v) >= 1e6:
        return f"{s}{v / 1e6:,.{decimals if decimals is not None else 1}f}M"
    if abs(v) >= 1e3:
        return f"{s}{v / 1e3:,.{decimals if decimals is not None else 1}f}K"
    return f"{s}{v:,.0f}"


def is_large_amount(value) -> bool:
    """True when a table value should be shown as a currency amount."""
    try:
        return abs(float(value)) >= _active["mid_div"]
    except (TypeError, ValueError):
        return False


def fmt_market_cap(value_in_large_units) -> str:
    """Format market cap given in the large unit (USD billions or rupee crore)."""
    if value_in_large_units is None:
        return "N/A"
    v = float(value_in_large_units)
    if get_currency() == "INR":
        return f"{symbol()}{v:,.0f} Cr"
    return f"{symbol()}{v:,.2f}B"


def amount_regex() -> str:
    """Regex for currency amounts in prose, used for auto-bolding."""
    return (r'(\$[\d,.]+\s*(?:billion|million|trillion|bn|mn|trn|B|M|K|T)'
            r'|(?:₹|Rs\.?\s?|INR\s?)[\d,.]+\s*(?:lakh\s+crore|crore|cr|lakh|bn|billion|mn|million|B|M)?)')


_STRIP = re.compile(r"[,$₹\s]|Rs\.?|INR|USD|Cr\b")


def parse_money(text) -> float | None:
    """Parse '$1,234.50' or '₹284.05' back to a float."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    cleaned = _STRIP.sub("", str(text))
    try:
        return float(cleaned)
    except ValueError:
        return None
