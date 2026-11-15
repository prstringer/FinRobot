#!/usr/bin/env python
# coding: utf-8
"""
Declarative chart specs (ECharts options as pure JSON) for the visual report, one function per chart.

``build_panel_charts(ctx, analysis_dir, kind)`` returns ``{panel_id: [Chart, ...]}`` following SCHEMA.md. ``ctx`` is
``report_context.json``; ``analysis_dir`` holds the CSVs written by the analysis step. Nothing is calculated for display
beyond what the charts themselves need (margins from the absolute rows, per-share values from the context's net debt and
share count); every plotted value is also written to the chart's ``datatable``.

A chart that cannot be built (missing input) is skipped and recorded in ``LAST_ERRORS``; the panel is still returned.
"""

from __future__ import annotations

import os
import statistics
from typing import Optional

import pandas as pd

from . import charts_common as cc
from .charts_common import (ACCENT, BG, GREY2, GREY3, GREY4, GREY5, GRID, MUTED, TEXT, fmt_cr, fmt_pct, fmt_rs, fmt_x,
                            fy, indian, make_chart, num, point, rnd)

PANEL_IDS = ["summary", "profit_engine", "cyclicality", "strength", "valuation", "price", "watch"]
LAST_ERRORS: list[str] = []
WINDOWS = {"6m": 126, "1y": 250, "2y": 500}
WINDOW_LABELS = {"6m": "6 months", "1y": "1 year", "2y": "2 years"}


# --------------------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------------------
def _csv(analysis_dir: str, name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(analysis_dir, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_fin_table(analysis_dir: str) -> Optional[dict]:
    """financial_metrics_and_forecasts.csv as {'years': [...], 'flags': [...], metric: [absolute values]}."""
    df = _csv(analysis_dir, "financial_metrics_and_forecasts.csv")
    if df is None or "metrics" not in df.columns:
        return None
    years = [c for c in df.columns if isinstance(c, str) and len(c) == 5 and c[:4].isdigit() and c[4] in "AE"]
    if not years:
        return None
    out = {"years": years, "flags": [cc.is_forecast(y) for y in years]}
    for _, r in df.iterrows():
        name = str(r["metrics"])
        key = "PE Ratio" if name.startswith("PE Ratio") else name
        out[key] = [num(r[y]) for y in years]
    return out


def _tag(ctx: dict) -> str:
    return str(ctx.get("meta", {}).get("ticker", "")).upper()


def _scale_cr(v: Optional[float]) -> Optional[float]:
    return None if v is None else v / 1e7


# --------------------------------------------------------------------------------------
# panel 2: the profit engine
# --------------------------------------------------------------------------------------
def margin_series(fin: dict) -> dict:
    rev, cp, sga, eb = fin["Revenue"], fin["Contribution Profit"], fin["SG&A"], fin["EBITDA"]

    def pct(a, b):
        return [None if x is None or y in (None, 0) else x / y * 100 for x, y in zip(a, b)]
    return {"cm": pct(cp, rev), "sga": pct(sga, rev), "em": pct(eb, rev)}


def chart_margin_scissors(ctx: dict, fin: dict) -> dict:
    years, flags = fin["years"], fin["flags"]
    labels = [fy(y) for y in years]
    m = margin_series(fin)
    cm, sga = m["cm"], m["sga"]
    actual_idx = [i for i, f in enumerate(flags) if not f]
    last_a, first_a = actual_idx[-1], actual_idx[0]
    gap = [None if a is None or b is None else a - b for a, b in zip(cm, sga)]

    def line(name, vals, color, annotate):
        a, f = cc.split_actual_forecast(vals, flags)
        a_items, f_items = [], []
        for i, v in enumerate(a):
            if v is None:
                a_items.append(None)
                continue
            lab = None
            if i == first_a:
                lab = f"{fmt_pct(v)}"
            if i == last_a:
                lab = annotate
            a_items.append(point(rnd(v, 2), tip=f"{labels[i]}: {name} {fmt_pct(v)}", label=lab,
                                 position="top" if name.startswith("Contribution") else "bottom",
                                 color=color if color != GREY5 else TEXT, bold=(i == last_a),
                                 offset=[-46, 6] if (i == last_a and not name.startswith("Contribution")) else None))
        for i, v in enumerate(f):
            if v is None or i == last_a:
                f_items.append(None if v is None else point(rnd(v, 2), tip=f"{labels[i]}: {name} {fmt_pct(v)}"))
                continue
            lab = fmt_pct(v) if i == len(f) - 1 else None
            f_items.append(point(rnd(v, 2), tip=f"{labels[i]} (forecast): {name} {fmt_pct(v)}", label=lab,
                                 position="top" if name.startswith("Contribution") else "bottom", color=TEXT))
        return a_items, f_items

    sga_act = [v for v, f in zip(sga, flags) if not f and v is not None]
    sga_last = sga[last_a]
    if sga_last >= max(sga_act) - 1e-9:
        sga_note = f"SG&A {fmt_pct(sga_last)}, five-year high"
    elif sga_last <= min(sga_act) + 1e-9:
        sga_note = f"SG&A {fmt_pct(sga_last)}, five-year low"
    else:
        sga_note = f"SG&A {fmt_pct(sga_last)}"
    cm_a, cm_f = line("Contribution margin", cm, GREY5, f"Contribution margin {fmt_pct(cm[last_a])}")
    sg_a, sg_f = line("SG&A (% of revenue)", sga, ACCENT, sga_note)

    lo = cc.nice_floor(min(v for v in sga if v is not None) - 5, 5)
    hi = cc.nice_ceil(max(v for v in cm if v is not None) + 5, 5)
    first_fc = next((labels[i] for i, f in enumerate(flags) if f), None)
    series = [
        {"name": "base", "type": "line", "stack": "gap", "data": [rnd(v, 2) for v in sga], "symbol": "none",
         "lineStyle": {"opacity": 0}, "silent": True, "tooltip": {"show": False}},
        {"name": "Gap between contribution margin and SG&A", "type": "line", "stack": "gap",
         "data": [point(rnd(g, 2), tip=f"{labels[i]}: gap {g:.1f} pts") for i, g in enumerate(gap)],
         "symbol": "none", "lineStyle": {"opacity": 0}, "areaStyle": {"color": GREY2, "opacity": 0.28}},
        {"name": "Contribution margin", "type": "line", "data": cm_a, "symbol": "circle", "symbolSize": 6,
         "lineStyle": {"color": GREY5, "width": 2.5}, "itemStyle": {"color": GREY5}},
        {"name": "Contribution margin (forecast)", "type": "line", "data": cm_f, "symbol": "circle", "symbolSize": 5,
         "lineStyle": cc.dashed_line(GREY5, 2), "itemStyle": {"color": GREY5}},
        {"name": "SG&A (% of revenue)", "type": "line", "data": sg_a, "symbol": "circle", "symbolSize": 6,
         "lineStyle": {"color": ACCENT, "width": 2.5}, "itemStyle": {"color": ACCENT}},
        {"name": "SG&A (% of revenue, forecast)", "type": "line", "data": sg_f, "symbol": "circle", "symbolSize": 5,
         "lineStyle": cc.dashed_line(ACCENT, 2), "itemStyle": {"color": ACCENT}},
    ]
    if first_fc:
        series[2]["markArea"] = cc.forecast_band(first_fc, labels[-1])
    option = cc.base_option(
        grid={"left": 48, "right": 24, "top": 34, "bottom": 28},
        xAxis=cc.cat_axis(labels, boundaryGap=True),
        yAxis=cc.val_axis(name=f"% of revenue (axis starts at {lo:.0f}%)", fmt="{value}%", min=lo, max=hi),
        series=series)
    rows = [[labels[i] + (" (forecast)" if flags[i] else ""), f"{cm[i]:.1f}", f"{sga[i]:.1f}", f"{gap[i]:.1f}"]
            for i in range(len(labels)) if cm[i] is not None and sga[i] is not None]
    a0, a1 = first_a, last_a
    alt = (f"SG&A moved from {sga[a0]:.1f}% to {sga[a1]:.1f}% of revenue between {labels[a0]} and {labels[a1]} "
           f"while contribution margin moved from {cm[a0]:.1f}% to {cm[a1]:.1f}%; the gap between the two lines "
           f"narrowed from {gap[a0]:.1f} to {gap[a1]:.1f} points. Forecasts are dashed.")
    return make_chart("margin_scissors", "main", "Prices held, costs rose", option, alt,
                      {"columns": ["Year", "Contribution margin (% of revenue)", "SG&A (% of revenue)", "Gap (points)"],
                       "rows": rows},
                      subtitle="Contribution margin against SG&A as % of revenue; shaded area is the gap", height=300)


def chart_margin_waterfall(ctx: dict, fin: dict) -> dict:
    years, flags = fin["years"], fin["flags"]
    actual = [i for i, f in enumerate(flags) if not f]
    last = actual[-1]
    m = margin_series(fin)
    pool = actual[-4:-1] or actual[:-1]
    best = max(pool, key=lambda i: m["em"][i] if m["em"][i] is not None else -1)
    pick = [best, last]
    yvals = []
    for i in pick:
        rev, cp, sg, eb = (_scale_cr(fin[k][i]) for k in ("Revenue", "Contribution Profit", "SG&A", "EBITDA"))
        yvals.append({"i": i, "rev": rev, "cogs": rev - cp, "cp": cp, "sga": sg, "eb": eb})
    ymax = cc.nice_ceil(max(v["rev"] for v in yvals) * 1.14, 1000)
    cats = ["Revenue", "Direct costs", "Contribution", "SG&A", "EBITDA"]
    grids, xs, ys, series, titles = [], [], [], [], []
    spans = [(6, 41), (55, 41)]
    for g, (v, (left, width)) in enumerate(zip(yvals, spans)):
        grids.append({"left": f"{left}%", "width": f"{width}%", "top": 64, "bottom": 30})
        xs.append(cc.cat_axis(cats, grid_index=g, axisLabel={"color": MUTED, "fontSize": 10, "interval": 0}))
        ys.append(cc.val_axis(grid_index=g, min=0, max=round(ymax), fmt="{value}",
                              **({} if g == 0 else {"show": False}),
                              **({"name": "Rs crore", "nameGap": 12} if g == 0 else {})))
        base = [0, v["cp"], 0, v["eb"], 0]
        vals = [v["rev"], v["cogs"], v["cp"], v["sga"], v["eb"]]
        labels = [fmt_cr(v["rev"]), "-" + fmt_cr(v["cogs"]), fmt_cr(v["cp"]), "-" + fmt_cr(v["sga"]), fmt_cr(v["eb"])]
        colors = [GREY3, GREY2, GREY4, ACCENT, GREY5]
        yl = fy(years[v["i"]])
        series.append({"name": f"{yl} base", "type": "bar", "stack": f"w{g}", "xAxisIndex": g, "yAxisIndex": g,
                       "data": base, "itemStyle": {"color": "transparent"}, "silent": True, "tooltip": {"show": False}})
        series.append({"name": f"{yl} bridge", "type": "bar", "stack": f"w{g}", "xAxisIndex": g, "yAxisIndex": g,
                       "barWidth": "58%",
                       "data": [{"value": rnd(vals[k], 1), "tip": f"{yl} {cats[k]}: {labels[k]}",
                                 "itemStyle": {"color": colors[k]},
                                 "label": {"show": True, "position": "top", "formatter": labels[k], "fontSize": 10,
                                           "color": TEXT, "fontWeight": "bold" if k in (3, 4) else "normal"}}
                                for k in range(5)]})
        tag = "best of the last three years" if g == 0 else "latest year"
        titles.append({"text": f"{yl} ({tag}): EBITDA margin {m['em'][v['i']]:.1f}%", "left": f"{left}%", "top": 4,
                       "textStyle": {"color": TEXT, "fontSize": 12, "fontWeight": "bold"}})
    option = cc.base_option(title=titles, grid=grids, xAxis=xs, yAxis=ys, series=series)
    rows = [[cats[k], f"{indian(yvals[0][key], 0)}", f"{indian(yvals[1][key], 0)}"]
            for k, key in enumerate(["rev", "cogs", "cp", "sga", "eb"])]
    y0, y1 = fy(years[pick[0]]), fy(years[pick[1]])
    alt = (f"From {y0} to {y1}, revenue went from {fmt_cr(yvals[0]['rev'])} to {fmt_cr(yvals[1]['rev'])}, SG&A from "
           f"{fmt_cr(yvals[0]['sga'])} to {fmt_cr(yvals[1]['sga'])}, and EBITDA from {fmt_cr(yvals[0]['eb'])} to "
           f"{fmt_cr(yvals[1]['eb'])}.")
    return make_chart("margin_waterfall", "small", "From revenue to EBITDA", option, alt,
                      {"columns": ["Step", f"{y0} (Rs crore)", f"{y1} (Rs crore)"], "rows": rows},
                      subtitle=f"{y0} is the highest EBITDA margin of the three years before {y1}; across all actual years "
                               f"the peak is {fy(years[max(actual, key=lambda i: m['em'][i] or -1)])}",
                      height=270)


# --------------------------------------------------------------------------------------
# panel 3: cyclicality
# --------------------------------------------------------------------------------------
def steadiest_peer(ctx: dict) -> Optional[dict]:
    ea = (ctx.get("peers") or {}).get("ebitda_actual_cr") or {}
    subject = _tag(ctx)
    series: dict[str, list[float]] = {}
    for y in sorted(ea):
        for t, v in ea[y].items():
            if v is not None and t.upper() != subject:
                series.setdefault(t, []).append(v)
    best = None
    for t, vals in series.items():
        if len(vals) >= 3 and statistics.mean(vals) > 0:
            cv = statistics.pstdev(vals) / statistics.mean(vals)
            if best is None or cv < best["cv"]:
                best = {"ticker": t, "cv": cv, "min": min(vals), "max": max(vals)}
    return best


def chart_cyclicality(ctx: dict, fin: dict) -> dict:
    years, flags = fin["years"], fin["flags"]
    labels = [fy(y) for y in years]
    growth = fin.get("Revenue Growth") or [None] * len(years)
    ebitda = [_scale_cr(v) for v in fin["EBITDA"]]
    peer = steadiest_peer(ctx)
    gmin = min([g for g in growth if g is not None] + [0])
    gmax = max([g for g in growth if g is not None] + [0])
    glo, ghi = cc.nice_floor(gmin - 6, 5), cc.nice_ceil(gmax + 6, 5)
    bars = []
    for i, g in enumerate(growth):
        if g is None:
            bars.append(None)
            continue
        base_col = GREY5 if g < 0 else GREY3
        style = cc.hatch(base_col) if flags[i] else {"color": base_col}
        sign = "-" if g < 0 else "+"
        bars.append({"value": rnd(g, 2), "tip": f"{labels[i]}{' (forecast)' if flags[i] else ''}: revenue growth {fmt_pct(g, 1, True)}",
                     "itemStyle": style,
                     "label": {"show": True, "position": "bottom" if g < 0 else "top",
                               "formatter": f"{sign}{abs(g):.1f}%", "fontSize": 11, "color": TEXT}})
    ea, ef = cc.split_actual_forecast(ebitda, flags)
    last_a = max(i for i, f in enumerate(flags) if not f)
    e_act = [None if v is None else point(rnd(v, 1), tip=f"{labels[i]}: EBITDA {fmt_cr(v)}",
                                          label=fmt_cr(v) if i in (0, last_a) or v == max(x for x in ea if x is not None) or
                                          v == min(x for x in ea if x is not None) else None, position="top")
             for i, v in enumerate(ea)]
    e_fc = [None if v is None or i == last_a else point(rnd(v, 1), tip=f"{labels[i]} (forecast): EBITDA {fmt_cr(v)}",
                                                      label=fmt_cr(v) if i == len(ef) - 1 else None)
            for i, v in enumerate(ef)]
    if last_a is not None and ef[last_a] is not None:
        e_fc[last_a] = point(rnd(ef[last_a], 1), tip=f"{labels[last_a]}: EBITDA {fmt_cr(ef[last_a])}")
    ymax = cc.nice_ceil(max([v for v in ebitda if v is not None] + ([peer["max"]] if peer else [])) * 1.18, 500)
    first_fc = next((labels[i] for i, f in enumerate(flags) if f), None)
    eb_series = {"name": "EBITDA", "type": "line", "xAxisIndex": 1, "yAxisIndex": 1, "data": e_act, "symbol": "circle",
                 "symbolSize": 6, "lineStyle": {"color": ACCENT, "width": 2.5}, "itemStyle": {"color": ACCENT}}
    if peer:
        eb_series["markArea"] = {"silent": True, "itemStyle": {"color": GREY2, "opacity": 0.3},
                                 "label": {"show": False},
                                 "data": [[{"yAxis": round(peer["min"], 1)}, {"yAxis": round(peer["max"], 1)}]]}
    fc_series = {"name": "EBITDA (forecast)", "type": "line", "xAxisIndex": 1, "yAxisIndex": 1, "data": e_fc,
                 "symbol": "circle", "symbolSize": 5, "lineStyle": cc.dashed_line(ACCENT, 2), "itemStyle": {"color": ACCENT}}
    if first_fc:
        fc_series["markArea"] = cc.forecast_band(first_fc, labels[-1])
    option = cc.base_option(
        title=[{"text": "Revenue growth, % a year", "left": 48, "top": 2, "textStyle": {"color": TEXT, "fontSize": 12}},
               {"text": "EBITDA, Rs crore" + (f" (shaded band: {peer['ticker']}, the steadiest peer, {fmt_cr(peer['min'])} to {fmt_cr(peer['max'])})" if peer else ""),
                "left": 48, "top": "52%", "textStyle": {"color": TEXT, "fontSize": 12}}],
        grid=[{"left": 48, "right": 24, "top": 26, "height": "32%"}, {"left": 48, "right": 24, "top": "60%", "bottom": 28}],
        xAxis=[cc.cat_axis(labels, grid_index=0, axisLabel={"show": False}, axisLine={"onZero": True, "lineStyle": {"color": GREY4}}),
               cc.cat_axis(labels, grid_index=1)],
        yAxis=[cc.val_axis(grid_index=0, min=glo, max=ghi, fmt="{value}%"),
               cc.val_axis(grid_index=1, min=0, max=ymax, fmt="{value}")],
        series=[{"name": "Revenue growth", "type": "bar", "xAxisIndex": 0, "yAxisIndex": 0, "data": bars, "barWidth": "52%"},
                eb_series, fc_series])
    rows = []
    for i, lab in enumerate(labels):
        rows.append([lab + (" (forecast)" if flags[i] else ""), "n/a" if growth[i] is None else f"{growth[i]:.1f}",
                     "n/a" if ebitda[i] is None else indian(ebitda[i], 0)])
    neg = [labels[i] for i, g in enumerate(growth) if g is not None and g < 0 and not flags[i]]
    act = [v for v, f in zip(ebitda, flags) if not f and v is not None]
    alt = (f"Revenue fell in {' and '.join(neg) if neg else 'no actual year'}"
           f" and EBITDA swung between {fmt_cr(min(act))} and {fmt_cr(max(act))} over {labels[0]} to {labels[last_a]}"
           + (f", while the steadiest peer, {peer['ticker']}, stayed between {fmt_cr(peer['min'])} and {fmt_cr(peer['max'])}."
              if peer else "."))
    cols = ["Year", "Revenue growth (%)", "EBITDA (Rs crore)"]
    return make_chart("cyclicality", "main", "Revenue growth and EBITDA through the cycle", option, alt,
                      {"columns": cols, "rows": rows},
                      subtitle="Top: revenue growth. Bottom: EBITDA, with the steadiest peer's range shaded", height=340)


# --------------------------------------------------------------------------------------
# panel 4: financial strength
# --------------------------------------------------------------------------------------
def chart_cash_conversion(ctx: dict) -> dict:
    cf = (ctx.get("extras") or {}).get("cashflow") or {}
    per = cf.get("per_year") or {}
    keys = [y for y in cf.get("years", []) if y in per]
    if cf.get("ttm_label") and cf["ttm_label"] in per:
        keys.append(cf["ttm_label"])
    labels = [fy(k) for k in keys]
    eb = [per[k].get("ebitda") for k in keys]
    cfo = [per[k].get("cfo") for k in keys]
    conv = [per[k].get("conversion") for k in keys]
    top = [max(a or 0, b or 0) for a, b in zip(eb, cfo)]
    ymax = cc.nice_ceil(max(top) * 1.2, 500)
    conv_pts = [point(rnd(t + ymax * 0.04, 1), label=("n/a" if c is None else f"{c * 100:.0f}%"), position="top", color=TEXT, bold=True,
                      tip=f"{labels[i]}: operating cash flow was {c * 100:.0f}% of EBITDA" if c is not None else None)
                for i, (t, c) in enumerate(zip(top, conv))]
    option = cc.base_option(
        tooltip={"trigger": "item"},
        legend={"top": 0, "right": 0, "textStyle": {"color": MUTED, "fontSize": 11}, "itemWidth": 12, "itemHeight": 10,
                "data": ["EBITDA", "Operating cash flow"]},
        grid={"left": 48, "right": 16, "top": 34, "bottom": 28},
        xAxis=cc.cat_axis(labels),
        yAxis=cc.val_axis(name="Rs crore", min=0, max=ymax),
        series=[
            {"name": "EBITDA", "type": "bar", "barGap": "8%", "barWidth": "30%", "itemStyle": {"color": GREY3},
             "data": [point(rnd(v, 1), tip=f"{labels[i]}: EBITDA {fmt_cr(v)}") for i, v in enumerate(eb)]},
            {"name": "Operating cash flow", "type": "bar", "barWidth": "30%", "itemStyle": {"color": ACCENT},
             "data": [point(rnd(v, 1), tip=f"{labels[i]}: operating cash flow {fmt_cr(v)}") for i, v in enumerate(cfo)]},
            {"name": "Cash conversion", "type": "line", "data": conv_pts, "symbol": "circle", "symbolSize": 6,
             "itemStyle": {"color": BG, "borderColor": BG}, "lineStyle": {"opacity": 0}, "silent": True, "z": 5}])
    rows = [[labels[i], fmt_cr(eb[i]), fmt_cr(cfo[i]), "n/a" if conv[i] is None else f"{conv[i] * 100:.0f}%"]
            for i in range(len(labels))]
    agg = (cf.get("aggregate") or {}).get("conversion")
    note = None
    if not cf.get("capex_available", True):
        note = ("Capex cannot be separated from investing cash flow in this data, so free cash flow is not shown. "
                "Investing cash flow is not presented as capex.")
    cyc = (cf.get("cyclicality") or {})
    alt = ("Operating cash flow as a share of EBITDA was " +
           ", ".join(f"{c * 100:.0f}% in {labels[i]}" for i, c in enumerate(conv) if c is not None and labels[i] != "TTM") +
           (f"; {agg * 100:.0f}% over the five years" if agg else "") +
           ("; cash conversion rose when revenue fell." if cyc.get("flag") else "."))
    return make_chart("cash_conversion", "main", "Does profit turn into cash?", option, alt,
                      {"columns": ["Year", "EBITDA", "Operating cash flow", "Cash conversion"], "rows": rows},
                      subtitle="EBITDA against operating cash flow; the percentage is operating cash flow divided by EBITDA. "
                               "TTM is the last four quarters.",
                      static_note=note, height=300)


def chart_debt_and_coverage(ctx: dict, analysis_dir: str) -> Optional[dict]:
    rb = (ctx.get("extras") or {}).get("ratio_bars") or {}
    ratios = _csv(analysis_dir, "ratios_raw_data.csv")
    specs = []
    for key, title, higher in (("debt_equity", "Debt to equity", False), ("interest_coverage", "Interest coverage", True)):
        r = rb.get(key)
        if not r or "scale" not in r:
            continue
        s = r["scale"]
        trend = ""
        if ratios is not None:
            col = {"debt_equity": "debtEquityRatio", "interest_coverage": "interestCoverage"}[key]
            if col in ratios.columns and "year" in ratios.columns:
                rr = ratios.sort_values("year")
                vals = rr[col].dropna()
                if len(vals) >= 2:
                    y0, y1 = int(rr.loc[vals.index[0], "year"]), int(rr.loc[vals.index[-1], "year"])
                    trend = f"{vals.iloc[0]:.2f}x in FY{str(y0)[2:]} to {vals.iloc[-1]:.2f}x in FY{str(y1)[2:]}"
        specs.append((key, title, higher, s, trend))
    if not specs:
        return None
    grids, xs, ys, series, titles, rows = [], [], [], [], [], []
    n = len(specs)
    for g, (key, title, higher, s, trend) in enumerate(specs):
        top = 8 + g * (100 // n)
        grids.append({"left": 24, "right": 32, "top": f"{top + 14}%", "height": "14%"})
        vmax = max(x for x in (s.get("range_max"), s.get("peer_median"), s.get("value")) if x is not None) * 1.2
        step = 0.2 if vmax < 2 else (1 if vmax < 10 else 2)
        xs.append(cc.val_axis(grid_index=g, min=0, max=cc.nice_ceil(vmax, step), fmt="{value}x", splitLine={"show": False}))
        ys.append(cc.cat_axis([""], grid_index=g, axisLine={"show": False}, axisLabel={"show": False}))
        lo, hi = s.get("range_min"), s.get("range_max")
        better = "higher is better" if higher else "lower is better"
        titles.append({"text": f"{title} ({better})" + (f": {trend}" if trend else ""), "left": 24, "top": f"{top}%",
                       "textStyle": {"color": TEXT, "fontSize": 12, "fontWeight": "bold"}})
        series.append({"name": f"{title} range base", "type": "bar", "stack": f"r{g}", "xAxisIndex": g, "yAxisIndex": g,
                       "data": [rnd(lo, 3)], "itemStyle": {"color": "transparent"}, "silent": True, "barWidth": 14,
                       "tooltip": {"show": False}})
        series.append({"name": f"{title}: own five-year range", "type": "bar", "stack": f"r{g}", "xAxisIndex": g,
                       "yAxisIndex": g, "barWidth": 14, "itemStyle": {"color": GREY2},
                       "data": [point(rnd(hi - lo, 3), tip=f"{title}: own five-year range {lo:.2f}x to {hi:.2f}x",
                                      label=f"range {lo:.2f}x to {hi:.2f}x", position="insideRight", color=MUTED)]})
        if s.get("peer_median") is not None:
            series.append({"name": f"{title}: peer median", "type": "scatter", "xAxisIndex": g, "yAxisIndex": g,
                           "symbol": "diamond", "symbolSize": 14, "itemStyle": {"color": GREY5},
                           "data": [point([rnd(s["peer_median"], 3), 0], tip=f"{title}: peer median {s['peer_median']:.2f}x",
                                          label=f"peer median {s['peer_median']:.2f}x", position="bottom", color=MUTED)]})
        series.append({"name": f"{title}: now", "type": "scatter", "xAxisIndex": g, "yAxisIndex": g, "symbolSize": 16,
                       "itemStyle": {"color": ACCENT, "borderColor": BG, "borderWidth": 2}, "z": 6,
                       "data": [point([rnd(s["value"], 3), 0], tip=f"{title}: now {s['value']:.2f}x",
                                      label=f"now {s['value']:.2f}x", position="top", color=TEXT, bold=True)]})
        rows.append([title, f"{s['value']:.2f}x", "n/a" if lo is None else f"{lo:.2f}x to {hi:.2f}x",
                     "n/a" if s.get("peer_median") is None else f"{s['peer_median']:.2f}x", better])
    option = cc.base_option(title=titles, grid=grids, xAxis=xs, yAxis=ys, series=series)
    def where(s):
        if s["value"] <= s["range_min"] + 1e-9:
            return "at the bottom of"
        if s["value"] >= s["range_max"] - 1e-9:
            return "at the top of"
        return "inside"
    alt = "; ".join(f"{t} is {s['value']:.2f}x now, {where(s)} its own five-year range of {s['range_min']:.2f}x to "
                    f"{s['range_max']:.2f}x" + (f" and against a peer median of {s['peer_median']:.2f}x" if s.get('peer_median') is not None else "")
                    for _, t, _, s, _ in specs) + "."
    return make_chart("debt_and_coverage", "small", "Debt and interest cover", option, alt,
                      {"columns": ["Ratio", "Now", "Own five-year range", "Peer median", "Direction"], "rows": rows},
                      subtitle="Dot: now. Bar: own five-year range. Diamond: peer median", height=250)


# --------------------------------------------------------------------------------------
# panel 5: valuation
# --------------------------------------------------------------------------------------
def _strip_series(points: list[dict], subject: dict, faded: Optional[dict], median: Optional[float],
                  median_label: str) -> list[dict]:
    peers = []
    for k, p in enumerate(sorted(points, key=lambda d: d["value"])):
        peers.append(point([rnd(p["value"], 2), 0], tip=p["tip"], label=f"{p['name']} {fmt_x(p['value'])}",
                           position="bottom", color=MUTED, offset=[0, 4 + (k % 2) * 16]))
    series = [{"name": "Peers", "type": "scatter", "symbolSize": 12, "itemStyle": {"color": GREY3}, "data": peers}]
    if median is not None:
        series[0]["markLine"] = {"silent": True, "symbol": "none", "lineStyle": {"color": GREY4, "type": "dashed", "width": 1.2},
                                 "label": {"show": True, "formatter": median_label, "position": "end", "color": MUTED,
                                           "fontSize": 10},
                                 "data": [{"xAxis": rnd(median, 2)}]}
    if faded:
        series.append({"name": faded["name"], "type": "scatter", "symbolSize": 20,
                       "itemStyle": {"color": ACCENT, "opacity": 0.3, "borderColor": ACCENT, "borderWidth": 1},
                       "data": [point([rnd(faded["value"], 2), 0], tip=faded["tip"], label=faded["label"],
                                      position="bottom", color=MUTED, offset=[0, 24])]})
    series.append({"name": subject["name"], "type": "scatter", "symbolSize": 24,
                   "itemStyle": {"color": ACCENT, "borderColor": BG, "borderWidth": 2}, "z": 6,
                   "data": [point([rnd(subject["value"], 2), 0], tip=subject["tip"], label=subject["label"],
                                  position="top", color=TEXT, bold=True)]})
    return series


def chart_valuation_strip(ctx: dict) -> Optional[dict]:
    peers = ctx.get("peers") or {}
    pc, pe = peers.get("current"), peers.get("ev_ebitda")
    subject = _tag(ctx)
    cur = ctx["current_valuation"]
    if not pc or not pe:
        return None
    rows_cur = pc["rows"]
    sub_row = next((r for t, r in rows_cur.items() if t.upper() == subject), {})
    avg_vals = {t: v for t, v in pe["values"].items() if v is not None}
    sub_avg = next((v for t, v in avg_vals.items() if t.upper() == subject), None)
    sub_ttm, sub_fy = sub_row.get("ev_ebitda_ttm"), sub_row.get("ev_ebitda_fy")
    price = ctx["market"]["price"]
    latest = ctx["meta"].get("latest_actual_year") or ""
    avg_price = (ctx.get("fy_series") or {}).get(latest, {}).get("avg_price")
    allv = [v for v in list(avg_vals.values()) + [r.get("ev_ebitda_ttm") for r in rows_cur.values()] +
            [r.get("ev_ebitda_fy") for r in rows_cur.values()] if v]
    axis_max = cc.nice_ceil(max(allv) * 1.12, 5)

    def variant(key: str) -> dict:
        if key == "today":
            pts = [{"name": t, "value": r["ev_ebitda_ttm"],
                    "tip": f"{t}: {fmt_x(r['ev_ebitda_ttm'])} on TTM EBITDA at today's price (Rs {r['price']:,.2f})"}
                   for t, r in rows_cur.items() if t.upper() != subject and r.get("ev_ebitda_ttm")]
            sub = {"name": subject, "value": sub_ttm,
                   "tip": f"{subject}: {fmt_x(sub_ttm)} on TTM EBITDA at today's price (Rs {price:,.2f})",
                   "label": f"{subject} {fmt_x(sub_ttm)} today"}
            faded = ({"name": "Old report multiple", "value": sub_avg,
                      "tip": f"{subject}: {fmt_x(sub_avg)} at the fiscal-year average price (the multiple the old report used)",
                      "label": f"old report {fmt_x(sub_avg)}"} if sub_avg else None)
            med = pc.get("median_excl_subject_ttm")
            lab = f"Peer median {fmt_x(med)}" if med else ""
            name = "EV/EBITDA, x (TTM EBITDA, today's price)"
        else:
            pts = [{"name": t, "value": v, "tip": f"{t}: {fmt_x(v)} on {pe['year']} EBITDA at its fiscal-year average price"}
                   for t, v in avg_vals.items() if t.upper() != subject]
            sub = {"name": subject, "value": sub_avg,
                   "tip": f"{subject}: {fmt_x(sub_avg)} on {pe['year']} EBITDA at the fiscal-year average price",
                   "label": f"{subject} {fmt_x(sub_avg)} at report price"}
            faded = ({"name": "Today", "value": sub_fy,
                      "tip": f"{subject}: {fmt_x(sub_fy)} on {pe['year']} EBITDA at today's price",
                      "label": f"today {fmt_x(sub_fy)}"} if sub_fy else None)
            med = pe.get("median_excl_subject")
            lab = f"Peer median {fmt_x(med)}" if med else ""
            name = f"EV/EBITDA, x ({pe['year']} EBITDA, fiscal-year average prices)"
        return cc.base_option(
            grid={"left": 24, "right": 36, "top": 64, "bottom": 40},
            xAxis=cc.val_axis(min=0, max=axis_max, fmt="{value}x", name=name, nameLocation="middle", nameGap=26,
                              nameTextStyle={"color": MUTED, "fontSize": 11}),
            yAxis=cc.cat_axis([""], axisLine={"show": False}, axisLabel={"show": False}),
            series=_strip_series(pts, sub, faded, med, lab))

    options = [{"key": "today", "label": "At today's price", "option": variant("today")},
               {"key": "report", "label": "At the report's price", "option": variant("report")}]
    note = None
    if avg_price and sub_avg:
        note = f"At the report's price of about Rs {avg_price:.0f}, EV/EBITDA was {fmt_x(sub_avg)}."
    rows = []
    for t in sorted(set(list(rows_cur) + list(avg_vals)), key=lambda t: (t.upper() != subject, t)):
        r = rows_cur.get(t, {})
        rows.append([t + (" (subject)" if t.upper() == subject else ""), fmt_x(r.get("ev_ebitda_ttm")), fmt_x(r.get("ev_ebitda_fy")),
                     fmt_x(avg_vals.get(t))])
    med_t, med_a = pc.get("median_excl_subject_ttm"), pe.get("median_excl_subject")
    rows.append(["Peer median (excl. subject)", fmt_x(med_t), fmt_x(pc.get("median_excl_subject_fy")), fmt_x(med_a)])
    ordered = sorted(((r["ev_ebitda_ttm"], t) for t, r in rows_cur.items() if r.get("ev_ebitda_ttm")))
    lo_p, hi_p = ordered[0], ordered[-1]
    alt = (f"At today's price of {fmt_rs(price)} {subject} trades at {fmt_x(sub_ttm)} TTM EV/EBITDA against a peer median of "
           f"{fmt_x(med_t)}" + (f" and the {fmt_x(sub_avg)} it showed at its fiscal-year average price" if sub_avg else "") +
           f"; peers range from {fmt_x(lo_p[0])} ({lo_p[1]}) to {fmt_x(hi_p[0])} ({hi_p[1]}).")
    return make_chart("valuation_strip", "main", "Where the multiple sits against peers", variant("today"), alt,
                      {"columns": ["Company", "TTM EBITDA, today's price", f"{ctx['meta'].get('latest_actual_year', 'FY')} EBITDA, today's price",
                                   "Fiscal-year average price"], "rows": rows},
                      subtitle="EV/EBITDA for each company; the dashed line is the peer median", variants=options,
                      default_variant="today", static_note=note, height=250)


def chart_pe_vs_history(ctx: dict) -> Optional[dict]:
    fys = ctx.get("fy_series") or {}
    latest = ctx["meta"].get("latest_actual_year")
    hist = {k: v["pe"] for k, v in sorted(fys.items()) if v.get("pe") and k != latest}
    cur = ctx["current_valuation"]
    if len(hist) < 2 or not (cur.get("pe_ttm_api") or cur.get("pe_ttm_computed")):
        return None
    lo, hi = min(hist.values()), max(hist.values())
    p_ttm = cur.get("pe_ttm_api") or cur.get("pe_ttm_computed")
    p_fy = cur.get("pe_latest_fy_at_current_price")
    price = ctx["market"]["price"]
    vals = [lo, hi, p_ttm] + ([p_fy] if p_fy else [])
    axis_max = cc.nice_ceil(max(vals) * 1.12, 10)
    yrs = f"{fy(next(iter(hist)))} to {fy(list(hist)[-1])}"
    series = [
        {"name": "base", "type": "bar", "stack": "b", "data": [rnd(lo, 2)], "itemStyle": {"color": "transparent"}, "silent": True,
         "barWidth": 18, "tooltip": {"show": False}},
        {"name": "Own P/E range", "type": "bar", "stack": "b", "barWidth": 18, "itemStyle": {"color": GREY2},
         "data": [point(rnd(hi - lo, 2), tip=f"Own P/E range {yrs}: {fmt_x(lo)} to {fmt_x(hi)}",
                        label=f"own range {yrs}: {fmt_x(lo)} to {fmt_x(hi)}", position="top", color=MUTED)]},
        {"name": "P/E on TTM EPS", "type": "scatter", "symbolSize": 18, "itemStyle": {"color": ACCENT, "borderColor": BG, "borderWidth": 2},
         "z": 6, "data": [point([rnd(p_ttm, 2), 0], tip=f"P/E on TTM EPS Rs {cur.get('eps_ttm')} at {fmt_rs(price)}: {fmt_x(p_ttm)}",
                                label=f"TTM EPS {cur.get('eps_ttm')}: {fmt_x(p_ttm)}", position="top", color=TEXT, bold=True)]}]
    if p_fy:
        series.append({"name": "P/E on last full-year EPS", "type": "scatter", "symbolSize": 18, "z": 6,
                       "itemStyle": {"color": BG, "borderColor": ACCENT, "borderWidth": 3},
                       "data": [point([rnd(p_fy, 2), 0], tip=f"P/E on {fy(latest)} EPS Rs {cur.get('eps_latest_fy')} at {fmt_rs(price)}: {fmt_x(p_fy)}",
                                      label=f"{fy(latest)} EPS {cur.get('eps_latest_fy')}: {fmt_x(p_fy)}", position="top", color=TEXT)]})
    option = cc.base_option(
        grid={"left": 24, "right": 36, "top": 48, "bottom": 40},
        xAxis=cc.val_axis(min=0, max=axis_max, fmt="{value}x", name=f"P/E, x (at today's price {fmt_rs(price)})",
                          nameLocation="middle", nameGap=26, nameTextStyle={"color": MUTED, "fontSize": 11}),
        yAxis=cc.cat_axis([""], axisLine={"show": False}, axisLabel={"show": False}), series=series)
    rows = [["Own range low", f"{yrs}", fmt_x(lo)], ["Own range high", f"{yrs}", fmt_x(hi)],
            ["P/E on TTM EPS, today's price", f"EPS Rs {cur.get('eps_ttm')}", fmt_x(p_ttm)]]
    if p_fy:
        rows.append([f"P/E on {fy(latest)} EPS, today's price", f"EPS Rs {cur.get('eps_latest_fy')}", fmt_x(p_fy)])
    alt = (f"P/E at today's price is {fmt_x(p_ttm)} on TTM EPS" + (f" and {fmt_x(p_fy)} on {fy(latest)} EPS" if p_fy else "") +
           f", against its own range of {fmt_x(lo)} to {fmt_x(hi)} from {yrs} (fiscal-year average prices).")
    return make_chart("pe_vs_history", "small", "P/E against its own history", option, alt,
                      {"columns": ["Measure", "Basis", "P/E"], "rows": rows},
                      subtitle="Band: own P/E range in earlier years. Filled dot: TTM EPS. Open dot: last full-year EPS",
                      height=190)


def per_share(multiple: float, ebitda_cr: float, net_debt_cr: float, shares_mn: float) -> float:
    return (multiple * ebitda_cr - net_debt_cr) * 10.0 / shares_mn


def chart_price_ruler(ctx: dict) -> Optional[dict]:
    fv = ctx.get("fair_value") or {}
    grid = fv.get("grid") or []
    mk, bal = ctx["market"], ctx["balance_latest"]
    price, shares, nd = mk.get("price"), mk.get("shares_mn"), bal.get("net_debt_cr")
    if not grid or not (price and shares) or nd is None:
        return None
    ebitda = fv["ebitda_cr"]
    y = fv.get("ebitda_year")
    mults = sorted(g["multiple"] for g in grid)
    cases = [(f"Our model: {mults[0]}x to {mults[-1]}x {fy(y)} EBITDA of {fmt_cr(ebitda)}",
              [per_share(m, ebitda, nd, shares) for m in (mults[0], mults[-1])])]
    tp = (ctx.get("peers") or {}).get("trend_projection")
    sub = _tag(ctx)
    peers_ok = not (ctx.get("peer_selection") or {}).get("insufficient")      # thin peer set: no peer-based fair-value cases
    if tp and peers_ok:
        e = next((v for t, v in tp["ebitda_cr"].items() if t.upper() == sub), None)
        if e:
            cases.append((f"Peer-dataset projection (not consensus): {mults[0]}x to {mults[-1]}x of {fmt_cr(e)}",
                          [per_share(m, e, nd, shares) for m in (mults[0], mults[-1])]))
    meds = [x for x in ((ctx["peers"].get("ev_ebitda") or {}).get("median_excl_subject"),
                        (ctx["peers"].get("current") or {}).get("median_excl_subject_ttm"),
                        (ctx["peers"].get("current") or {}).get("median_excl_subject_fy")) if x]
    if meds and peers_ok:
        cases.append((f"Peer median multiple ({fmt_x(min(meds))} to {fmt_x(max(meds))}) on our {fy(y)} EBITDA",
                      [per_share(min(meds), ebitda, nd, shares), per_share(max(meds), ebitda, nd, shares)]))
    axis_max = cc.nice_ceil(max(max(c[1]) for c in cases + [("", [price])]) * 1.15, 50)
    cats = [c[0] for c in cases]
    n = len(cases)
    row_h, top = 56, 34
    base = [rnd(min(c[1]), 1) for c in cases]
    rows, items = [], []
    for name, (a_, b_) in cases:
        lo, hi = min(a_, b_), max(a_, b_)
        d_lo, d_hi = (lo / price - 1) * 100, (hi / price - 1) * 100
        lab = f"{fmt_rs(lo, 0)} to {fmt_rs(hi, 0)} ({d_lo:+.0f}% to {d_hi:+.0f}%)"
        items.append(point(rnd(hi - lo, 1), tip=f"{name}: {lab} against today's price", label=lab, position="left", color=TEXT))
        rows.append([name, fmt_rs(lo, 0), fmt_rs(hi, 0), f"{d_lo:+.0f}% to {d_hi:+.0f}%"])
    titles = [{"text": c, "left": 12, "top": top + k * row_h + 2, "textStyle": {"color": MUTED, "fontSize": 11, "fontWeight": "normal"}}
              for k, c in enumerate(cats)]
    option = cc.base_option(
        title=titles,
        grid={"left": 12, "right": 60, "top": top, "height": row_h * n},
        xAxis=cc.val_axis(min=0, max=axis_max, fmt="Rs {value}", name="Value per share, Rs", nameLocation="middle", nameGap=26,
                          nameTextStyle={"color": MUTED, "fontSize": 11}),
        yAxis=cc.cat_axis(cats, inverse=True, axisLabel={"show": False}, axisLine={"show": False}),
        series=[{"name": "base", "type": "bar", "stack": "r", "data": base, "itemStyle": {"color": "transparent"}, "silent": True,
                 "barWidth": 20, "tooltip": {"show": False}},
                {"name": "Fair value per share", "type": "bar", "stack": "r", "barWidth": 20, "itemStyle": {"color": GREY3},
                 "data": items,
                 "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": ACCENT, "type": "solid", "width": 2.5},
                              "label": {"show": True, "formatter": f"Today {fmt_rs(price)}", "position": "start", "color": TEXT,
                                        "fontSize": 11, "fontWeight": "bold"},
                              "data": [{"xAxis": rnd(price, 2)}]}}])
    lo_all, hi_all = min(min(c[1]) for c in cases), max(max(c[1]) for c in cases)
    m_lo, m_hi = min(cases[0][1]), max(cases[0][1])
    rel = "below" if m_hi < price else ("above" if m_lo > price else "around")
    alt = (f"Today's price is {fmt_rs(price)}. Fair-value bands per share run from {fmt_rs(lo_all, 0)} to {fmt_rs(hi_all, 0)} across "
           f"{n} cases; the model case is {fmt_rs(m_lo, 0)} to {fmt_rs(m_hi, 0)}, "
           f"{abs((m_hi / price - 1) * 100):.0f}% to {abs((m_lo / price - 1) * 100):.0f}% {rel} today's price.")
    return make_chart("price_ruler", "small", "Price against fair-value bands", option, alt,
                      {"columns": ["Case", "Low (Rs per share)", "High (Rs per share)", "Versus today's price"], "rows": rows},
                      subtitle="Per share = (multiple x EBITDA - net debt) / shares; the line is today's price",
                      height=top + row_h * n + 44)


# --------------------------------------------------------------------------------------
# panel 6: price behaviour
# --------------------------------------------------------------------------------------
def _prepared_prices(analysis_dir: str) -> Optional[pd.DataFrame]:
    df = _csv(analysis_dir, "historical_price_full.csv")
    if df is None or len(df) < 60:
        return None
    from modules import technical_analysis as ta
    return ta.prepare(df)


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _dlabel(s: str) -> str:
    t = pd.Timestamp(s)
    return f"{t.day} {t.strftime('%b %y')}"


def _in_window(date: str, win: pd.DataFrame) -> bool:
    return win["date"].iloc[0] <= pd.Timestamp(date) <= win["date"].iloc[-1]


def _row_at(win: pd.DataFrame, date: str):
    r = win[win["date"] == pd.Timestamp(date)]
    return None if r.empty else r.iloc[0]


def chart_price_trend(ctx: dict, prep: pd.DataFrame, tf: dict) -> dict:
    from modules import technical_analysis as ta
    div = (tf.get("divergence") or {})
    cross = tf.get("crossover_50_200")
    try:
        t_idx, _ = ta._trend_window(prep)
        trend_start = prep["date"].iloc[t_idx]
    except Exception:
        trend_start = None

    def option_for(n: int) -> dict:
        win = prep.tail(n).reset_index(drop=True)
        dates = [_d(x) for x in win["date"]]
        close = [[d, rnd(v, 2)] for d, v in zip(dates, win["close"])]
        s50 = [[d, rnd(v, 2)] for d, v in zip(dates, win["sma50"]) if pd.notna(v)]
        s200 = [[d, rnd(v, 2)] for d, v in zip(dates, win["sma200"]) if pd.notna(v)]
        vol = [[d, rnd(v / 1e6, 3)] for d, v in zip(dates, win["volume"])]
        hi_i = int(win["close"].idxmax())
        pts = [{"coord": [dates[hi_i], rnd(win["close"].iloc[hi_i], 2)], "value": "", "symbol": "circle", "symbolSize": 9,
                "itemStyle": {"color": ACCENT},
                "label": {"show": True, "formatter": f"Latest high {fmt_rs(win['close'].iloc[hi_i])}, {_dlabel(dates[hi_i])}",
                          "position": "top", "color": TEXT, "fontSize": 11}}]
        if trend_start is not None and win["date"].iloc[0] <= trend_start < win["date"].iloc[-1]:
            r = _row_at(win, _d(trend_start))
            if r is not None:
                pts.append({"coord": [_d(trend_start), rnd(r["close"], 2)], "value": "", "symbol": "circle", "symbolSize": 9,
                            "itemStyle": {"color": GREY5},
                            "label": {"show": True, "formatter": f"Trend start {_dlabel(_d(trend_start))}", "position": "bottom",
                                      "color": MUTED, "fontSize": 11}})
        if cross and _in_window(cross["date"], win):
            r = _row_at(win, cross["date"])
            if r is not None and pd.notna(r["sma50"]):
                pts.append({"coord": [cross["date"], rnd(r["sma50"], 2)], "value": "", "symbol": "diamond", "symbolSize": 14,
                            "itemStyle": {"color": GREY5},
                            "label": {"show": True, "formatter": f"{cross['direction'].capitalize()} cross {_dlabel(cross['date'])}",
                                      "position": "bottom", "color": TEXT, "fontSize": 11}})
        lines = []
        for d in (div.get("bearish") or [])[:1] + (div.get("bullish") or [])[:1]:
            if _in_window(d["price_date_1"], win):
                lines.append([{"coord": [d["price_date_1"], d["price_1"]], "lineStyle": {"color": TEXT, "type": "dashed", "width": 1.5},
                               "symbol": "circle"},
                              {"coord": [d["price_date_2"], d["price_2"]], "symbol": "arrow",
                               "label": {"show": True, "formatter": "Higher high" if d["kind"] == "bearish" else "Lower low",
                                         "position": "start", "color": TEXT, "fontSize": 10}}])
        main = {"name": "Close", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0, "data": close, "showSymbol": False,
                "lineStyle": {"color": ACCENT, "width": 1.8}, "itemStyle": {"color": ACCENT}, "z": 5,
                "endLabel": {"show": True, "formatter": f"Close {fmt_rs(win['close'].iloc[-1])}", "color": TEXT, "fontSize": 11},
                "markPoint": {"silent": True, "data": pts}}
        if lines:
            main["markLine"] = {"silent": True, "data": lines, "label": {"show": True}}
        return cc.base_option(
            tooltip={"trigger": "axis", "axisPointer": {"type": "cross"}},
            axisPointer={"link": [{"xAxisIndex": "all"}]},
            grid=[{"left": 52, "right": 120, "top": 34, "height": "56%"}, {"left": 52, "right": 120, "top": "78%", "height": "12%"}],
            xAxis=[{"type": "time", "gridIndex": 0, "axisLabel": {"show": False}, "axisTick": {"show": False},
                    "axisLine": {"lineStyle": {"color": GRID}}, "splitLine": {"show": False}},
                   {"type": "time", "gridIndex": 1, "axisTick": {"show": False}, "axisLine": {"lineStyle": {"color": GRID}},
                    "axisLabel": {"color": MUTED, "fontSize": 11}, "splitLine": {"show": False}}],
            yAxis=[cc.val_axis(grid_index=0, scale=True, name="Rs per share", fmt="{value}"),
                   cc.val_axis(grid_index=1, name="Volume, million shares", fmt="{value}", splitNumber=2,
                               axisLabel={"color": MUTED, "fontSize": 9, "formatter": "{value}"})],
            series=[main,
                    {"name": "50-day average", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0, "data": s50, "showSymbol": False,
                     "lineStyle": {"color": GREY3, "width": 1.5}, "endLabel": {"show": True, "formatter": "50-day", "color": MUTED, "fontSize": 10}},
                    {"name": "200-day average", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0, "data": s200, "showSymbol": False,
                     "lineStyle": {"color": GREY5, "width": 1.5}, "endLabel": {"show": True, "formatter": "200-day", "color": MUTED, "fontSize": 10}},
                    {"name": "Volume", "type": "bar", "xAxisIndex": 1, "yAxisIndex": 1, "data": vol, "barMaxWidth": 4,
                     "itemStyle": {"color": GREY2}}])

    variants = [{"key": k, "label": WINDOW_LABELS[k], "option": option_for(n)} for k, n in WINDOWS.items()]
    win = prep.tail(WINDOWS["1y"]).reset_index(drop=True)
    rows = [[_d(r.date), f"{r.close:.2f}", "n/a" if pd.isna(r.sma50) else f"{r.sma50:.2f}",
             "n/a" if pd.isna(r.sma200) else f"{r.sma200:.2f}", f"{int(r.volume):,}"] for r in win.itertuples()]
    first, last = win.iloc[0], win.iloc[-1]
    alt = (f"Price went from {fmt_rs(first['close'])} on {_dlabel(_d(first['date']))} to {fmt_rs(last['close'])} on "
           f"{_dlabel(_d(last['date']))}, {tf.get('price_vs_sma200_pct', 0):.0f}% above its 200-day average"
           + (f"; the 50-day average crossed {'above' if cross['direction'] == 'golden' else 'below'} the 200-day on {_dlabel(cross['date'])}"
              if cross else "")
           + f"; the latest session's volume was {tf.get('volume_latest_vs_20d', 0):.2f}x its 20-day average.")
    return make_chart("price_trend", "main", "Price, moving averages and volume", variants[1]["option"], alt,
                      {"columns": ["Date", "Close (Rs)", "50-day average", "200-day average", "Volume (shares)"], "rows": rows},
                      subtitle="Daily close with the 50-day and 200-day averages; volume beneath",
                      variants=variants, default_variant="1y",
                      static_note="Shown: 1 year. The 6-month and 2-year views are available in the on-screen report.", height=360)


def chart_momentum(ctx: dict, prep: pd.DataFrame, tf: dict) -> dict:
    div = tf.get("divergence") or {}

    def option_for(n: int) -> dict:
        win = prep.tail(n).reset_index(drop=True)
        dates = [_d(x) for x in win["date"]]
        macd = [[d, rnd(v, 3)] for d, v in zip(dates, win["macd"])]
        sig = [[d, rnd(v, 3)] for d, v in zip(dates, win["macd_signal"])]
        hist = [[d, rnd(v, 3)] for d, v in zip(dates, win["macd_hist"])]
        rsi = [[d, rnd(v, 2)] for d, v in zip(dates, win["rsi"]) if pd.notna(v)]
        mlines, rlines = [], []
        for d in (div.get("bearish") or []) + (div.get("bullish") or []):
            if not _in_window(d["indicator_date_1"], win):
                continue
            line = [{"coord": [d["indicator_date_1"], d["indicator_1"]], "symbol": "circle",
                     "lineStyle": {"color": TEXT, "type": "dashed", "width": 1.5}},
                    {"coord": [d["indicator_date_2"], d["indicator_2"]], "symbol": "arrow",
                     "label": {"show": True, "position": "end", "color": TEXT, "fontSize": 10,
                               "formatter": f"{d['indicator']} {d['indicator_1']:.1f} to {d['indicator_2']:.1f}"}}]
            (mlines if d["indicator"] == "MACD" else rlines).append(line)
        macd_s = {"name": "MACD", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0, "data": macd, "showSymbol": False,
                  "lineStyle": {"color": ACCENT, "width": 1.6}, "itemStyle": {"color": ACCENT}, "z": 4}
        if mlines:
            macd_s["markLine"] = {"silent": True, "data": mlines, "label": {"show": True}}
        rsi_s = {"name": "RSI (14)", "type": "line", "xAxisIndex": 1, "yAxisIndex": 1, "data": rsi, "showSymbol": False,
                 "lineStyle": {"color": GREY5, "width": 1.5}, "itemStyle": {"color": GREY5},
                 "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": GREY3, "type": "dashed", "width": 1},
                              "label": {"show": True, "position": "insideEndTop", "color": MUTED, "fontSize": 10},
                              "data": [{"yAxis": 70, "label": {"formatter": "70 overbought"}},
                                       {"yAxis": 30, "label": {"formatter": "30 oversold"}}] + (rlines and []) }}
        if rlines:
            rsi_s["markLine"]["data"] = rsi_s["markLine"]["data"] + rlines
        return cc.base_option(
            tooltip={"trigger": "axis", "axisPointer": {"type": "cross"}},
            axisPointer={"link": [{"xAxisIndex": "all"}]},
            title=[{"text": "MACD (line, signal, histogram)", "left": 52, "top": 0, "textStyle": {"color": TEXT, "fontSize": 12}},
                   {"text": "RSI (14)", "left": 52, "top": "58%", "textStyle": {"color": TEXT, "fontSize": 12}}],
            grid=[{"left": 52, "right": 120, "top": 22, "height": "34%"}, {"left": 52, "right": 120, "top": "66%", "height": "20%"}],
            xAxis=[{"type": "time", "gridIndex": 0, "axisLabel": {"show": False}, "axisTick": {"show": False},
                    "axisLine": {"lineStyle": {"color": GRID}}, "splitLine": {"show": False}},
                   {"type": "time", "gridIndex": 1, "axisTick": {"show": False}, "axisLine": {"lineStyle": {"color": GRID}},
                    "axisLabel": {"color": MUTED, "fontSize": 11}, "splitLine": {"show": False}}],
            yAxis=[cc.val_axis(grid_index=0, scale=True, fmt="{value}"), cc.val_axis(grid_index=1, min=0, max=100, fmt="{value}", splitNumber=2)],
            series=[{"name": "Histogram", "type": "bar", "xAxisIndex": 0, "yAxisIndex": 0, "data": hist, "barMaxWidth": 4,
                     "itemStyle": {"color": GREY2}},
                    {"name": "Signal", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0, "data": sig, "showSymbol": False,
                     "lineStyle": {"color": GREY4, "width": 1.2}},
                    macd_s, rsi_s])

    variants = [{"key": k, "label": WINDOW_LABELS[k], "option": option_for(n)} for k, n in WINDOWS.items()]
    win = prep.tail(WINDOWS["1y"]).reset_index(drop=True)
    rows = [[_d(r.date), f"{r.macd:.3f}", f"{r.macd_signal:.3f}", f"{r.macd_hist:.3f}", "n/a" if pd.isna(r.rsi) else f"{r.rsi:.1f}"]
            for r in win.itertuples()]
    b = (div.get("bearish") or [None])[0]
    macd_pk = tf.get("macd_peak_90d"), tf.get("macd_peak_90d_date")
    if b:
        alt = (f"Price made a higher high ({fmt_rs(b['price_1'])} on {_dlabel(b['price_date_1'])} to {fmt_rs(b['price_2'])} on "
               f"{_dlabel(b['price_date_2'])}) while MACD made a lower high ({b['indicator_1']:.2f} to {b['indicator_2']:.2f}): a bearish divergence. "
               f"MACD is {tf.get('macd'):.2f} now, RSI {tf.get('rsi14'):.1f}, with a 90-day RSI peak of {tf.get('rsi_peak_90d'):.1f}.")
    else:
        alt = (f"No divergence between price and MACD or RSI at the last two swing points. MACD is {tf.get('macd'):.2f} now "
               f"(90-day peak {macd_pk[0]}), RSI {tf.get('rsi14'):.1f}.")
    return make_chart("momentum", "small", "Momentum: MACD and RSI", variants[1]["option"], alt,
                      {"columns": ["Date", "MACD", "Signal", "Histogram", "RSI (14)"], "rows": rows},
                      subtitle="Dashed lines join the last two highs where the indicator disagrees with price",
                      variants=variants, default_variant="1y",
                      static_note="Shown: 1 year. The 6-month and 2-year views are available in the on-screen report.", height=300)


# --------------------------------------------------------------------------------------
# panel 1: price says / earnings say
# --------------------------------------------------------------------------------------
def chart_price_vs_earnings(ctx: dict) -> Optional[dict]:
    pve = ((ctx.get("extras") or {}).get("technical") or {}).get("price_vs_earnings") or {}
    if pve.get("price_change_1y_pct") is None:
        return None
    items = [("Share price, 1 year", pve["price_change_1y_pct"], ACCENT),
             ("EPS, last full year", pve.get("eps_fy_change_pct"), GREY4),
             ("TTM EPS vs last full year", pve.get("eps_ttm_change_pct"), GREY3)]
    items = [i for i in items if i[1] is not None]
    vals = [v for _, v, _ in items]
    lo, hi = cc.nice_floor(min(vals + [0]) - 20, 20), cc.nice_ceil(max(vals + [0]) + 20, 20)
    data = [{"value": rnd(v, 1), "tip": f"{n}: {fmt_pct(v, 0, True)}", "itemStyle": {"color": c},
             "label": {"show": True, "position": "right" if v >= 0 else "left", "formatter": fmt_pct(v, 0, True), "color": TEXT,
                       "fontSize": 12, "fontWeight": "bold"}} for n, v, c in items]
    option = cc.base_option(
        grid={"left": 200, "right": 56, "top": 8, "bottom": 28},
        xAxis=cc.val_axis(min=lo, max=hi, fmt="{value}%"),
        yAxis=cc.cat_axis([n for n, _, _ in items], inverse=True, axisLabel={"color": TEXT, "fontSize": 12},
                          axisLine={"lineStyle": {"color": GREY4}}),
        series=[{"name": "Change", "type": "bar", "barWidth": 18, "data": data}])
    rows = [[n, fmt_pct(v, 1, True)] for n, v, _ in items]
    alt = (f"The share price moved {fmt_pct(pve['price_change_1y_pct'], 0, True)} in a year; EPS moved "
           f"{fmt_pct(pve.get('eps_fy_change_pct'), 0, True)} in the last full year and TTM EPS is "
           f"{fmt_pct(pve.get('eps_ttm_change_pct'), 0, True)} against {pve.get('eps_ttm_base_label', 'the last full year')}.")
    return make_chart("price_vs_earnings", "small", "Price says / earnings say", option, alt,
                      {"columns": ["Measure", "Change"], "rows": rows},
                      subtitle=pve.get("sentence"), height=150)


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def _try(name: str, fn, out: list) -> None:
    try:
        chart = fn()
    except Exception as e:           # keep the panel; record why a chart is missing
        LAST_ERRORS.append(f"{name}: {type(e).__name__}: {e}")
        return
    if chart:
        out.append(chart)
    else:
        LAST_ERRORS.append(f"{name}: skipped (missing input)")


def build_panel_charts(ctx: dict, analysis_dir: str, kind: str = "nonfinancial") -> dict[str, list[dict]]:
    """Charts for each panel id. Bank variants are added later: kind='bank' returns every panel with no charts."""
    LAST_ERRORS.clear()
    panels: dict[str, list[dict]] = {pid: [] for pid in PANEL_IDS}
    if kind == "bank":
        return panels
    fin = load_fin_table(analysis_dir)
    if fin and all(k in fin for k in ("Revenue", "Contribution Profit", "SG&A", "EBITDA")):
        _try("margin_scissors", lambda: chart_margin_scissors(ctx, fin), panels["profit_engine"])
        _try("margin_waterfall", lambda: chart_margin_waterfall(ctx, fin), panels["profit_engine"])
        _try("cyclicality", lambda: chart_cyclicality(ctx, fin), panels["cyclicality"])
    else:
        LAST_ERRORS.append("profit_engine/cyclicality: financial_metrics_and_forecasts.csv missing or incomplete")
    _try("cash_conversion", lambda: chart_cash_conversion(ctx), panels["strength"])
    _try("debt_and_coverage", lambda: chart_debt_and_coverage(ctx, analysis_dir), panels["strength"])
    _try("valuation_strip", lambda: chart_valuation_strip(ctx), panels["valuation"])
    ps_ = ctx.get("peer_selection") or {}
    if ps_.get("insufficient"):            # a thin peer set is shown, but never as a valuation benchmark
        for ch in panels["valuation"]:
            if ch.get("id") == "valuation_strip":
                ch["subtitle"] = (f"Thin peer set ({len(ps_.get('peers') or [])} comparable peers within the market-cap band and sub-sector): "
                                  "indicative only, not a valuation benchmark. " + (ch.get("subtitle") or "")).strip()
    _try("pe_vs_history", lambda: chart_pe_vs_history(ctx), panels["valuation"])
    _try("price_ruler", lambda: chart_price_ruler(ctx), panels["valuation"])
    prep = None
    try:
        prep = _prepared_prices(analysis_dir)
    except Exception as e:
        LAST_ERRORS.append(f"price: {type(e).__name__}: {e}")
    if prep is not None:
        from modules import technical_analysis as ta
        pve = ((ctx.get("extras") or {}).get("technical") or {}).get("price_vs_earnings") or {}
        info = {"eps_fy_now": pve.get("eps_fy_now"), "eps_fy_prev": pve.get("eps_fy_prev"), "eps_ttm": pve.get("eps_ttm")}
        tf = ta.compute_technical_facts(prep[["date", "open", "high", "low", "close", "volume"]], info)
        if tf.get("available", True):
            _try("price_trend", lambda: chart_price_trend(ctx, prep, tf), panels["price"])
            _try("momentum", lambda: chart_momentum(ctx, prep, tf), panels["price"])
    else:
        LAST_ERRORS.append("price: historical_price_full.csv missing or too short")
    _try("price_vs_earnings", lambda: chart_price_vs_earnings(ctx), panels["summary"])
    return panels
