#!/usr/bin/env python
# coding: utf-8
"""
HTML renderer for the visual report.

Reads ONLY the report model (see SCHEMA.md) and writes ONE self-contained HTML file: inline CSS, JS, the vendored
ECharts build and the model JSON. No network access. All model strings are HTML-escaped (Jinja autoescape); nothing is
calculated here beyond layout helpers (Indian number grouping, scale-bar positions, first-use glossary links).

* ``render_html_string(model, print_mode=False)`` returns the page. ``print_mode=True`` produces the A4 print layout:
  light theme, no navigation or toggles, one panel per page, datatables open (up to ``MAX_PRINT_ROWS`` rows), running
  header and footer, ``data-print-panel`` on each panel, ``window.__reportReady`` and ``window.__printOverflow``.
* ``render_html(model, out_path)`` writes the screen version.

Model fields relied on beyond SCHEMA.md: see ``REQUIRED_FIELDS`` and the notes at the bottom of this docstring.
Optional, tolerated when absent: ``panel.extra.divergence.sentence|text``, ``panel.extra.readings``, ``panel.extra.rsi_zone``,
``panel.extra.price``, ``warnings[].level`` in {error, warning, info}, ``sources.provider|snapshot|definitions|validation``.
"""

from __future__ import annotations

import copy
import functools
import html as _html
import json
import os
import re
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(HERE, "templates")
ASSET_DIR = os.path.join(HERE, "assets")
MAX_PRINT_ROWS = 40          # datatables longer than this stay closed in print (their values are drawn as chart labels)
EM_DASH = "\u2014"

REQUIRED_FIELDS = ("meta", "panels")
VERDICT_ICON = {"good": "check", "neutral": "dash", "concern": "alert"}
VERDICT_WORD = {"good": "Good", "neutral": "Neutral", "concern": "Concern"}
SOURCE_TEXT = {"report_data": "report data", "calculated": "calculated in this report", "general_norm": "general norm, not company-specific"}
LVL_WORD = {"error": "Error", "warning": "Warning", "info": "Note"}


# ----------------------------------------------------------------------------------------------
# helpers (formatting only)
# ----------------------------------------------------------------------------------------------
def inr(value: Any, dec: int = 0) -> str:
    """Indian digit grouping: 12,34,567.89. None or non-numeric -> 'n/a'."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    neg = v < 0
    s = f"{abs(v):.{dec}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    out = whole + ("." + frac if frac else "")
    return ("-" if neg else "") + out


def signed(value: Any, dec: int = 0) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{v:+.{dec}f}"


def scale_pos(scale: Optional[dict], value: Any) -> Optional[float]:
    """Position (0 to 100) of a value inside the scale's min..max, clamped. None when the scale is incomplete."""
    if not scale or value is None:
        return None
    lo, hi = scale.get("min"), scale.get("max")
    if lo is None or hi is None:
        return None
    try:
        lo, hi, v = float(lo), float(hi), float(value)
    except (TypeError, ValueError):
        return None
    if hi == lo:
        return 50.0
    return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100.0))


def scale_text(value: Any, unit: Optional[str] = None) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    suffix = unit if unit in ("%", "x") else ""
    return f"{v:,.1f}{suffix}"


def scale_aria(scale: dict, verdict: dict) -> str:
    bits = [f"Range {scale_text(scale.get('min'), verdict.get('unit'))} to {scale_text(scale.get('max'), verdict.get('unit'))}",
            f"current {verdict.get('value_text')}"]
    if scale.get("peer_median") is not None:
        bits.append(f"peer median {scale_text(scale.get('peer_median'), verdict.get('unit'))}")
    return "; ".join(bits)


class GlossaryLinker:
    """Wraps the first use of each glossary term, across the whole document, in a link to its definition."""

    def __init__(self, glossary: list[dict]):
        self.ids: dict[str, str] = {}
        terms = []
        for g in glossary or []:
            term = str(g.get("term", "")).strip()
            if term and g.get("id"):
                esc = str(escape(term))
                self.ids[esc.lower()] = g["id"]
                terms.append(esc)
        terms.sort(key=len, reverse=True)
        self.used: set[str] = set()
        self.rx = re.compile(r"(?<![\w/&])(" + "|".join(re.escape(t) for t in terms) + r")(?![\w/])", re.I) if terms else None

    def __call__(self, text: Any) -> Markup:
        if text is None:
            return Markup("")
        esc = str(escape(str(text)))
        if not self.rx:
            return Markup(esc)

        def link(m: re.Match) -> str:
            gid = self.ids.get(m.group(1).lower())
            if not gid or gid in self.used:
                return m.group(0)
            self.used.add(gid)
            return f'<a class="gl" href="#gl-{gid}">{m.group(0)}</a>'
        return Markup(self.rx.sub(link, esc))


# ----------------------------------------------------------------------------------------------
# model preparation
# ----------------------------------------------------------------------------------------------
def _clean_strings(obj: Any) -> Any:
    """Deep copy with em-dashes replaced (the report never emits them)."""
    if isinstance(obj, str):
        return re.sub(r"\s*" + EM_DASH + r"\s*", ", ", obj)
    if isinstance(obj, list):
        return [_clean_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean_strings(v) for k, v in obj.items()}
    return obj


def prepare_model(model: dict) -> dict:
    for f in REQUIRED_FIELDS:
        if f not in model:
            raise ValueError(f"report model is missing required field '{f}'")
    m = _clean_strings(copy.deepcopy(model))
    if not m["panels"]:
        raise ValueError("report model has no panels")
    m["panels"] = sorted(m["panels"], key=lambda p: p.get("n", 0))
    for p in m["panels"]:
        for f in ("id", "headline"):
            if not p.get(f):
                raise ValueError(f"panel {p.get('n')} is missing '{f}'")
        p.setdefault("charts", [])
        p.setdefault("chips", [])
        p.setdefault("notes", [])
        p.setdefault("extra", {})
        for c in p["charts"]:
            c.setdefault("role", "main")
    m.setdefault("glossary", [])
    m.setdefault("warnings", [])
    m.setdefault("sources", {})
    m.setdefault("disclaimer", "")
    return m


@functools.lru_cache(maxsize=1)
def _echarts_js() -> str:
    with open(os.path.join(ASSET_DIR, "echarts.min.js"), encoding="utf-8") as fh:
        s = fh.read()
    return s.replace(EM_DASH, "\\u2014").replace("</script", "<\\/script")


def _asset(name: str) -> str:
    with open(os.path.join(ASSET_DIR, name), encoding="utf-8") as fh:
        return fh.read()


def _embed_json(model: dict) -> str:
    s = json.dumps(model, ensure_ascii=True, separators=(",", ":"))
    # \u003c etc. are valid JSON escapes: no raw '<', '>' or '&' can ever reach the HTML parser from model data
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(enabled_extensions=("html", "j2"), default_for_string=True),
        trim_blocks=True, lstrip_blocks=True,
    )
    env.filters.update({"inr": inr, "signed": signed, "scale_pos": scale_pos, "scale_text": scale_text, "scale_aria": scale_aria})
    return env


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------
def render_html_string(model: dict, print_mode: bool = False) -> str:
    m = prepare_model(model)
    tpl = _env().get_template("report.html.j2")
    page = tpl.render(
        model=m, meta=m["meta"], panels=m["panels"], glossary=m["glossary"], sources=m["sources"],
        print_mode=print_mode, gl=GlossaryLinker(m["glossary"]), max_print_rows=MAX_PRINT_ROWS,
        verdict_icon=VERDICT_ICON, verdict_word=VERDICT_WORD, source_text=SOURCE_TEXT, lvl_word=LVL_WORD,
        css=_asset("report.css"), js=_asset("report.js"), echarts_js=_echarts_js(), model_json=_embed_json(m),
    )
    return page.replace(EM_DASH, ", ")


def render_html(model: dict, out_path: str) -> str:
    """Write the screen version (self-contained HTML) and return its path."""
    page = render_html_string(model, print_mode=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    return out_path


if __name__ == "__main__":   # python -m report_visual.render_html model.json out.html [--print]
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        raise SystemExit("usage: render_html.py MODEL.json OUT.html [--print]")
    with open(args[0], encoding="utf-8") as fh:
        mdl = json.load(fh)
    if "--print" in sys.argv:
        out = render_html_string(mdl, print_mode=True)
        with open(args[1], "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out)
        print(args[1])
    else:
        print(render_html(mdl, args[1]))
