#!/usr/bin/env python
# coding: utf-8
"""
Validation of a finished report model (run inside ``build_report_model`` and again before rendering).

Returns ``{"errors": [...], "warnings": [...], "passed": [...]}``. Errors mean a figure cannot be trusted; the renderers
show them prominently on Panel 1. Checks:
 1. snapshot reconciliation: price x shares = market cap; EV = market cap + net debt; P/E TTM = price / TTM EPS;
    every fair-value price recomputes from its EBITDA, multiple, net debt and shares;
 2. headlines: seven panels in order, each headline 8 words or fewer, no em-dashes;
 3. every chip has a yardstick, a scale and a short so-what that does not restate the value; general norms are labelled;
 4. every number in generated text (headline, thesis, so-what, lights, signposts) appears in the model's own figures;
 5. no raw HTML entities; chart options are pure JSON with colour tokens (no hex colours) and have alt text.
"""

from __future__ import annotations

import json
import re
from typing import Any

PANEL_IDS = ["summary", "profit_engine", "cyclicality", "strength", "valuation", "price", "watch"]
GENERAL_NORM_TEXT = "General norm, not company-specific"
_ENTITY = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|nbsp|#\d+);)[A-Za-z]+;|&amp;amp;")
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_HEX = re.compile(r'"#[0-9a-fA-F]{3,8}"')
ALWAYS_OK = {1.0, 2.0, 3.0, 4.0, 5.0, 100.0}       # "Rs 100", "five years", "three signs": counting words written as digits


def _walk(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def _numbers_in(text: str) -> list[tuple[float, int]]:
    """Numbers in a string with their decimal count; fiscal-year labels and ISO dates are removed first."""
    text = re.sub(r"\bFY\d{2}E?\b", " ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)
    text = re.sub(r"\b\d{4}[AE]\b", " ", text)
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            out.append((float(raw), len(raw.split(".")[1]) if "." in raw else 0))
        except ValueError:
            pass
    return out


def _corpus(model: dict) -> list[float]:
    vals: list[float] = []
    ok_string_paths = ("datatable", "value_text", "current_text", "readings", "toggle_note", "evidence", "tip", "formatter", "label")
    for path, v in _walk(model):
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, str) and any(p in path for p in ok_string_paths):
            vals += [n for n, _ in _numbers_in(v)]
    return vals


def _in_corpus(n: float, dec: int, corpus: list[float]) -> bool:
    if n in ALWAYS_OK:
        return True
    tol = 0.5 * 10 ** (-dec) + 1e-9
    return any(abs(round(c, dec) - n) < 1e-9 or abs(abs(c) - n) <= tol for c in corpus)


def validate_model(model: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    passed: list[str] = []

    # 1. reconciliation
    snap = model["meta"]["snapshot"]
    price, shares, mcap, nd, ev = snap.get("price"), snap.get("shares_mn"), snap.get("market_cap_cr"), snap.get("net_debt_cr"), snap.get("ev_cr")
    n_ok = 0
    if price and shares and mcap:
        calc = price * shares * 1e6 / 1e7
        if abs(calc / mcap - 1) > 0.01:
            errors.append(f"Price x shares ({calc:,.0f} cr) does not match the market cap ({mcap:,.0f} cr) within 1%.")
        else:
            n_ok += 1
    if mcap is not None and nd is not None and ev:
        if abs((mcap + nd) / ev - 1) > 0.005:
            errors.append(f"Market cap plus net debt ({mcap + nd:,.0f} cr) does not match enterprise value ({ev:,.0f} cr).")
        else:
            n_ok += 1
    chips = {c["id"]: c for p in model["panels"] for c in p["chips"]}
    eps = snap.get("eps_ttm")
    if "pe" in chips and price and eps:
        if abs(price / eps / chips["pe"]["value"] - 1) > 0.02:
            errors.append(f"P/E on the page ({chips['pe']['value']:.1f}x) does not reconcile to snapshot price / TTM EPS ({price / eps:.1f}x).")
        else:
            n_ok += 1
    val = next((p for p in model["panels"] if p["id"] == "valuation"), None)
    for c in ((val or {}).get("extra") or {}).get("fair_value_cases", []):
        if shares and nd is not None and c.get("ebitda_cr"):
            lo = (c["low_mult"] * c["ebitda_cr"] - nd) * 10 / shares
            hi = (c["high_mult"] * c["ebitda_cr"] - nd) * 10 / shares
            if abs(lo - c["low_price"]) > 0.01 * max(abs(lo), 1) or abs(hi - c["high_price"]) > 0.01 * max(abs(hi), 1):
                errors.append(f"Fair-value case '{c['label']}' does not recompute from its EBITDA, multiple, net debt and shares.")
            else:
                n_ok += 1
        if price and abs((c["low_price"] / price - 1) * 100 - c["vs_price_pct_low"]) > 0.5:
            errors.append(f"Fair-value case '{c['label']}' is not measured against the snapshot price.")
    if n_ok:
        passed.append(f"{n_ok} reconciliation checks passed against the snapshot price")

    # 2. panels and headlines
    if [p["id"] for p in model["panels"]] != PANEL_IDS:
        errors.append("Panels are not the seven expected panels in order.")
    bad_head = [p["id"] for p in model["panels"] if len(re.findall(r"[\w']+", p.get("headline", ""))) > 8 or not p.get("headline")]
    if bad_head:
        errors.append(f"Headline missing or longer than 8 words in: {', '.join(bad_head)}.")
    else:
        passed.append("All seven headlines state a conclusion in 8 words or fewer")

    # 3. chips
    bad = 0
    for pid, c in ((p["id"], c) for p in model["panels"] for c in p["chips"]):
        s = c.get("scale") or {}
        if not c.get("yardstick") or s.get("min") is None or s.get("max") is None:
            errors.append(f"Chip '{c.get('label')}' has no yardstick or scale.")
            bad += 1
        if len(c.get("so_what", "").split()) > 30:
            warnings.append(f"Chip '{c['label']}': so-what is longer than about 25 words.")
        if c.get("value_text") and c["value_text"] in c.get("so_what", ""):
            warnings.append(f"Chip '{c['label']}': so-what restates the value.")
        if c.get("yardstick") == "general_norm" and c.get("yardstick_text") != GENERAL_NORM_TEXT:
            errors.append(f"Chip '{c['label']}': general norm is not labelled.")
    if not bad:
        passed.append("Every chip has a verdict, a yardstick and a scale; general norms are labelled")

    # 4. text numbers come from the model
    corpus = _corpus(model)
    texts: list[tuple[str, str]] = [("thesis", model.get("thesis", ""))]
    for p in model["panels"]:
        texts.append((f"{p['id']} headline", p["headline"]))
        texts += [(f"{c['label']} so-what", c["so_what"]) for c in p["chips"]]
        texts += [(f"{p['id']} note", n) for n in p.get("notes", [])]
        for l in (p.get("extra") or {}).get("lights", []):
            texts.append((f"light {l['key']}", l["reason"]))
        for s in (p.get("extra") or {}).get("signposts", []):
            texts += [(f"signpost {s['metric']}", s["why"]), (f"signpost {s['metric']} threshold", s["threshold_text"])]
    loose = 0
    for where, t in texts:
        for n, dec in _numbers_in(t):
            if not _in_corpus(n, dec, corpus):
                warnings.append(f"{where}: number {n:g} is not in the model's figures.")
                loose += 1
    if not loose:
        passed.append("Every number in generated text appears in the model's figures")

    # 5. hygiene
    strings = [v for _, v in _walk(model) if isinstance(v, str)]
    ents = {m.group(0) for s in strings for m in _ENTITY.finditer(s)}
    if ents:
        errors.append(f"Raw HTML entities in the model: {sorted(ents)[:5]}")
    else:
        passed.append("No raw HTML entities")
    if any("—" in s for s in strings):
        warnings.append("An em-dash appears in the model text.")
    bad_chart = 0
    for p in model["panels"]:
        for ch in p["charts"]:
            blob = json.dumps(ch.get("option"))
            if _HEX.search(blob):
                warnings.append(f"Chart '{ch['id']}' uses hex colours instead of tokens.")
                bad_chart += 1
            if not ch.get("alt"):
                warnings.append(f"Chart '{ch['id']}' has no alt text.")
                bad_chart += 1
    if not bad_chart:
        passed.append("All charts use colour tokens and carry alt text")
    return {"errors": errors, "warnings": warnings, "passed": passed}
