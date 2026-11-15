#!/usr/bin/env python
# coding: utf-8
"""
Shared helpers for the visual-report chart specs (ECharts options as pure JSON).

* colour TOKENS (never hex), fiscal-year labels, Indian number formats,
* forecast styling (dashed lines, hatched bars, a labelled "Forecast" band),
* ``validate_option``: JSON-serialisable, no functions, no hex/rgb colours, animation off, no pie/radar/3D,
* ``make_chart``: the Chart envelope defined in SCHEMA.md (option, alt, datatable, variants, static note, height).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

# --------------------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------------------
TOKENS = {"@good", "@neutral", "@concern", "@accent", "@grey1", "@grey2", "@grey3", "@grey4", "@grey5",
          "@text", "@muted", "@grid", "@bg"}
# verdict colours are reserved for verdict chips; data charts must not use them
VERDICT_TOKENS = {"@good", "@neutral", "@concern"}

ACCENT, GREY1, GREY2, GREY3, GREY4, GREY5 = "@accent", "@grey1", "@grey2", "@grey3", "@grey4", "@grey5"
TEXT, MUTED, GRID, BG = "@text", "@muted", "@grid", "@bg"

_COLOR_KEYS = {"color", "borderColor", "backgroundColor", "shadowColor", "textBorderColor", "textShadowColor",
               "areaColor", "fill", "stroke"}
_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_FORBIDDEN_TYPES = {"pie", "radar", "funnel", "sunburst", "treemap", "sankey", "gauge", "bar3D", "line3D", "scatter3D",
                    "surface", "pictorialBar"}
_NON_COLOURS = {"", "transparent", "none", "inherit", "auto"}


# --------------------------------------------------------------------------------------
# labels and number formats (all pre-formatted in Python)
# --------------------------------------------------------------------------------------
def fy(label: str) -> str:
    """'2026A' -> 'FY26', '2027E' -> 'FY27E', 'TTM' -> 'TTM', '2026' -> 'FY26'."""
    s = str(label)
    m = re.fullmatch(r"(\d{4})([AE]?)", s)
    if not m:
        return s
    return f"FY{m.group(1)[2:]}{'E' if m.group(2) == 'E' else ''}"


def is_forecast(label: str) -> bool:
    return str(label).endswith("E") and bool(re.fullmatch(r"\d{4}E", str(label)))


def indian(n: float, dec: int = 0) -> str:
    """Indian digit grouping: 1234567.8 -> '12,34,567' (dec=0)."""
    neg = n < 0
    s = f"{abs(n):.{dec}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        whole = f"{head},{tail}"
    out = whole + (f".{frac}" if frac else "")
    return f"-{out}" if neg else out


def fmt_cr(v: Optional[float], dec: Optional[int] = None) -> str:
    """Rs 9,399 cr (one decimal below Rs 100 cr)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    d = dec if dec is not None else (0 if abs(v) >= 100 else 1)
    return f"Rs {indian(v, d)} cr"


def fmt_rs(v: Optional[float], dec: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"Rs {indian(v, dec)}"


def fmt_pct(v: Optional[float], dec: int = 1, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:+.{dec}f}%" if sign else f"{v:.{dec}f}%"


def fmt_x(v: Optional[float], dec: int = 1) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{dec}f}x"


def num(x: Any) -> Optional[float]:
    """float or None for NaN/None/'N/A'/'8.4%'."""
    if x is None:
        return None
    try:
        v = float(str(x).replace("%", "").replace(",", "").strip())
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def rnd(v: Optional[float], d: int = 2) -> Optional[float]:
    return None if v is None else round(float(v), d)


def nice_floor(v: float, step: float) -> float:
    return math.floor(v / step) * step


def nice_ceil(v: float, step: float) -> float:
    return math.ceil(v / step) * step


# --------------------------------------------------------------------------------------
# option building blocks
# --------------------------------------------------------------------------------------
def base_option(**extra) -> dict:
    opt = {"animation": False, "textStyle": {"fontSize": 11, "color": TEXT}, "tooltip": {"trigger": "item"}}
    opt.update(extra)
    return opt


def cat_axis(labels: list, grid_index: int = 0, **kw) -> dict:
    ax = {"type": "category", "data": labels, "gridIndex": grid_index, "axisTick": {"show": False},
          "axisLine": {"lineStyle": {"color": GRID}}, "axisLabel": {"color": MUTED, "fontSize": 11}}
    ax.update(kw)
    return ax


def val_axis(grid_index: int = 0, name: Optional[str] = None, fmt: str = "{value}", **kw) -> dict:
    ax = {"type": "value", "gridIndex": grid_index, "splitLine": {"lineStyle": {"color": GRID, "type": "dashed"}},
          "axisLine": {"show": False}, "axisTick": {"show": False},
          "axisLabel": {"color": MUTED, "fontSize": 11, "formatter": fmt}}
    if name:
        ax.update({"name": name, "nameTextStyle": {"color": MUTED, "fontSize": 10, "align": "left"}})
    ax.update(kw)
    return ax


def hatch(color: str = GREY3) -> dict:
    """Hatched bar style for forecasts: solid colour with diagonal lines in the background colour."""
    return {"color": color, "decal": {"symbol": "rect", "color": BG, "dashArrayX": [1, 0], "dashArrayY": [2, 4],
                                       "rotation": -0.785, "symbolSize": 1}}


def dashed_line(color: str, width: float = 2) -> dict:
    return {"color": color, "width": width, "type": "dashed"}


def forecast_band(first_label: str, last_label: str) -> dict:
    """markArea shading the forecast years, labelled 'Forecast'."""
    return {"silent": True, "itemStyle": {"color": GREY2, "opacity": 0.16},
            "label": {"show": True, "position": "insideTop", "color": MUTED, "fontSize": 10},
            "data": [[{"name": "Forecast", "xAxis": first_label}, {"xAxis": last_label}]]}


def split_actual_forecast(values: list, flags: list[bool]) -> tuple[list, list]:
    """Two aligned series: actuals, and forecasts that start at the last actual point so the line joins."""
    actual = [v if not f else None for v, f in zip(values, flags)]
    last_a = max((i for i, f in enumerate(flags) if not f), default=None)
    fc = [None] * len(values)
    if last_a is not None:
        fc[last_a] = values[last_a]
    for i, f in enumerate(flags):
        if f:
            fc[i] = values[i]
    return actual, fc


def point(value, tip: Optional[str] = None, label: Optional[str] = None, position: str = "top", color: str = TEXT,
          bold: bool = False, offset: Optional[list] = None, **extra) -> dict:
    """Data item with a tooltip string and an optional direct label."""
    d: dict = {"value": value}
    if tip:
        d["tip"] = tip
    if label is not None:
        d["label"] = {"show": True, "formatter": label, "position": position, "color": color, "fontSize": 11,
                      "fontWeight": "bold" if bold else "normal"}
        if offset:
            d["label"]["offset"] = offset
    d.update(extra)
    return d


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------
def _walk(node: Any, path: str, problems: list[str]) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}"
            if k in _COLOR_KEYS:
                _check_colour(v, p, problems)
            if k == "type" and isinstance(v, str) and v in _FORBIDDEN_TYPES and "series" in path:
                problems.append(f"{p}: series type '{v}' is not allowed")
            if k in ("radar", "polar", "grid3D", "xAxis3D", "geo"):
                problems.append(f"{p}: '{k}' is not allowed")
            _walk(v, p, problems)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk(v, f"{path}[{i}]", problems)
    elif isinstance(node, str):
        if _HEX.search(node) and not node.startswith("@"):
            problems.append(f"{path}: hex colour in string '{node[:30]}'")
        if node.strip().startswith("function") or "=>" in node:
            problems.append(f"{path}: looks like JavaScript")
        if re.search(r"rgba?\(", node):
            problems.append(f"{path}: rgb()/rgba() colour; use a token")


def _check_colour(v: Any, path: str, problems: list[str]) -> None:
    if isinstance(v, str):
        if v in _NON_COLOURS:
            return
        if v not in TOKENS:
            problems.append(f"{path}: colour '{v}' is not a token")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            _check_colour(x, f"{path}[{i}]", problems)
    elif isinstance(v, dict):
        pass     # gradient objects: walk() visits their colour keys
    elif v is not None and not isinstance(v, (int, float)):
        problems.append(f"{path}: unexpected colour value {v!r}")


def _uses_verdict_tokens(node: Any) -> bool:
    return any(t in json.dumps(node) for t in (f'"{x}"' for x in VERDICT_TOKENS))


def validate_option(option: dict, allow_verdict_tokens: bool = False) -> dict:
    """Assert an option is renderer-safe. Returns it unchanged; raises ValueError listing every problem."""
    problems: list[str] = []
    try:
        json.dumps(option, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"option is not JSON-serialisable: {e}") from e
    if option.get("animation") is not False:
        problems.append("animation must be false")
    _walk(option, "option", problems)
    if not allow_verdict_tokens and _uses_verdict_tokens(option):
        problems.append("verdict colour tokens (@good/@neutral/@concern) are reserved for verdict chips")
    if problems:
        raise ValueError("invalid chart option: " + "; ".join(problems))
    return option


# --------------------------------------------------------------------------------------
# chart envelope
# --------------------------------------------------------------------------------------
def make_chart(id: str, role: str, title: str, option: dict, alt: str, datatable: dict, subtitle: Optional[str] = None,
               variants: Optional[list] = None, default_variant: Optional[str] = None, static_note: Optional[str] = None,
               height: int = 260) -> dict:
    assert role in ("main", "small"), role
    validate_option(option)
    if variants:
        for v in variants:
            validate_option(v["option"])
        if default_variant is None:
            default_variant = variants[0]["key"]
        if not any(v["key"] == default_variant for v in variants):
            raise ValueError("default_variant is not among the variants")
    assert datatable and datatable.get("columns") and datatable.get("rows") is not None, "datatable required"
    width = len(datatable["columns"])
    for r in datatable["rows"]:
        assert len(r) == width and all(isinstance(c, str) for c in r), f"datatable row malformed in {id}: {r}"
    return {"id": id, "role": role, "title": title, "subtitle": subtitle, "option": option, "alt": alt,
            "datatable": datatable, "variants": variants or None, "default_variant": default_variant if variants else None,
            "static_note": static_note, "height": height}
