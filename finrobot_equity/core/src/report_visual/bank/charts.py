#!/usr/bin/env python
# coding: utf-8
"""
Bank chart specs (ECharts options, pure JSON, colour tokens) following SCHEMA.md and the chart rules in charts_common.

``build_bank_panel_charts(ctx, analysis_dir)`` returns ``{panel_id: [Chart, ...]}``; which charts a panel carries is read from
``variants/bank.yaml``. The price-behaviour charts are shared with the non-financial report. A chart that cannot be built
is skipped and recorded in ``LAST_ERRORS``.
"""

from __future__ import annotations

from typing import Optional

from .. import charts as nf
from .. import charts_common as cc
from ..charts_common import ACCENT, BG, GREY2, GREY3, GREY4, GREY5, MUTED, TEXT, fmt_rs, fmt_x, fy, make_chart, point, rnd
from . import providers as P

LAST_ERRORS: list[str] = []
PANEL_IDS = nf.PANEL_IDS


def pct(v: Optional[float], d: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{d}f}%"


def _years(ctx: dict, name: str) -> tuple[list[str], list[float]]:
    s = P.actual_series(ctx, name)
    return list(s.keys()), list(s.values())


def _tag(ctx: dict) -> str:
    return ctx["meta"]["ticker"].upper()


# --------------------------------------------------------------------------------------
# panel 2: how the bank earns
# --------------------------------------------------------------------------------------
def chart_income_split(ctx: dict) -> Optional[dict]:
    ti, opex, prov, pat = (P.actual_series(ctx, n) for n in ("Total Income", "Operating Expenses", "Provisions", "Net Profit"))
    years = [y for y in ti if y in opex and y in prov and y in pat and ti[y]]
    if len(years) < 2:
        return None
    labels = [fy(y) for y in years]
    cost = [opex[y] / ti[y] * 100 for y in years]
    pro = [prov[y] / ti[y] * 100 for y in years]
    prof = [pat[y] / ti[y] * 100 for y in years]
    rest = [100 - a - b - c for a, b, c in zip(cost, pro, prof)]

    def seg(name, vals, color, show_label):
        data = [point(rnd(v, 1), tip=f"{lab}: {name} Rs {v:.0f} of every Rs 100 of income",
                      label=f"{v:.0f}" if show_label else None, position="inside", color=BG if color in (ACCENT, GREY5) else TEXT, bold=show_label)
                for lab, v in zip(labels, vals)]
        return {"name": name, "type": "bar", "stack": "income", "barWidth": 44, "itemStyle": {"color": color}, "data": data}
    series = [seg("Operating costs", cost, GREY5, True), seg("Provisions for bad loans", pro, GREY3, True),
              seg("Tax and other", rest, GREY2, False), seg("Profit", prof, ACCENT, True)]
    option = cc.base_option(
        grid={"left": 40, "right": 16, "top": 24, "bottom": 36},
        xAxis=cc.cat_axis(labels), yAxis=cc.val_axis(min=0, max=100, fmt="{value}", name="Rs of each Rs 100 of income", interval=25),
        series=series)
    rows = [[lab, f"{c:.1f}", f"{p:.1f}", f"{r:.1f}", f"{f:.1f}", cc.fmt_cr(ti[y] / 1e7)] for lab, y, c, p, r, f in zip(labels, years, cost, pro, rest, prof)]
    alt = (f"Of every Rs 100 of income, {labels[-1]} cost Rs {cost[-1]:.0f} to run, Rs {pro[-1]:.0f} went to provisions and Rs {prof[-1]:.0f} was kept as profit; "
           f"in {labels[0]} the bank kept Rs {prof[0]:.0f}.")
    return make_chart("income_split", "main", "Where each Rs 100 of income goes", option, alt,
                      {"columns": ["Year", "Operating costs", "Provisions", "Tax and other", "Profit", "Total income"], "rows": rows},
                      subtitle="Stacked: costs, provisions, tax and other, profit. Dark grey: costs; accent: profit", height=290)


def chart_nim_trend(ctx: dict) -> Optional[dict]:
    years, vals = _years(ctx, "NIM")
    if len(vals) < 3:
        return None
    labels = [fy(y) for y in years]
    med = ((ctx["peers"].get("bank") or {}).get("median_excl_subject") or {}).get("nim")
    lo, hi = min(vals + ([med] if med else [])), max(vals + ([med] if med else []))
    data = [point(rnd(v, 2), tip=f"{lab}: net interest margin {pct(v)}", label=pct(v), position="top", color=TEXT, bold=(i == len(vals) - 1))
            for i, (lab, v) in enumerate(zip(labels, vals))]
    line = {"name": "Net interest margin", "type": "line", "data": data, "symbolSize": 8, "lineStyle": {"color": ACCENT, "width": 3},
            "itemStyle": {"color": ACCENT}}
    if med:
        line["markLine"] = {"silent": True, "symbol": "none", "lineStyle": {"color": GREY4, "type": "dashed", "width": 1.2},
                            "label": {"show": True, "formatter": f"Peer median {pct(med)}", "position": "insideEndTop", "color": MUTED, "fontSize": 10},
                            "data": [{"yAxis": rnd(med, 2)}]}
    option = cc.base_option(grid={"left": 40, "right": 24, "top": 28, "bottom": 28}, xAxis=cc.cat_axis(labels),
                            yAxis=cc.val_axis(min=cc.nice_floor(lo - 0.3, 0.5), max=cc.nice_ceil(hi + 0.3, 0.5), fmt="{value}%"), series=[line])
    rows = [[l, pct(v)] for l, v in zip(labels, vals)] + ([["Peer median (latest year)", pct(med)]] if med else [])
    alt = (f"Net interest margin moved from {pct(vals[0])} in {labels[0]} to {pct(vals[-1])} in {labels[-1]}"
           + (f", against a peer median of {pct(med)}." if med else "."))
    return make_chart("nim_trend", "small", "Net interest margin", option, alt, {"columns": ["Year", "NIM"], "rows": rows},
                      subtitle="Line: the bank. Dashed: peer median", height=190)


# --------------------------------------------------------------------------------------
# panel 3: growth and cycle
# --------------------------------------------------------------------------------------
def chart_growth_cycle(ctx: dict) -> Optional[dict]:
    adv, dep, pro = (P.actual_series(ctx, n) for n in ("Advances Growth", "Deposits Growth", "Net Profit Growth"))
    years = [y for y in pro if y in adv and y in dep]
    if len(years) < 2:
        return None
    labels = [fy(y) for y in years]
    cap = 40.0
    bars = []
    for name, s, color in (("Loan growth", adv, GREY4), ("Deposit growth", dep, GREY2)):
        data = []
        for y, lab in zip(years, labels):
            v = s[y]
            merged = name == "Loan growth" and v > cap
            tip = f"{lab}: {name.lower()} {v:.1f}%" + (" (merger-affected, not like-for-like)" if merged else "")
            item = point(rnd(v, 1), tip=tip, label=(f"{v:.0f}% merger" if merged else f"{v:.0f}%"), position="top", color=MUTED)
            if merged:
                item["itemStyle"] = cc.hatch(color)
            data.append(item)
        bars.append({"name": name, "type": "bar", "barGap": "10%", "barWidth": 22, "itemStyle": {"color": color}, "data": data})
    line = {"name": "Net profit growth", "type": "line", "symbolSize": 9, "lineStyle": {"color": ACCENT, "width": 3}, "itemStyle": {"color": ACCENT},
            "data": [point(rnd(pro[y], 1), tip=f"{lab}: net profit growth {pro[y]:.1f}%", label=f"{pro[y]:.1f}%", position="top", color=TEXT, bold=True)
                     for y, lab in zip(years, labels)], "z": 5}
    vmax = max(max(adv[y], dep[y], pro[y]) for y in years)
    vmin = min(0, min(min(adv[y], dep[y], pro[y]) for y in years))
    option = cc.base_option(grid={"left": 40, "right": 16, "top": 28, "bottom": 36}, xAxis=cc.cat_axis(labels),
                            yAxis=cc.val_axis(min=cc.nice_floor(vmin, 10), max=cc.nice_ceil(vmax * 1.1, 10), fmt="{value}%"),
                            series=bars + [line])
    rows = [[l, f"{adv[y]:.1f}", f"{dep[y]:.1f}", f"{pro[y]:.1f}"] for l, y in zip(labels, years)]
    gv = [pro[y] for y in years]
    alt = (f"Net profit growth ran {', '.join(f'{v:.0f}%' for v in gv)} from {labels[0]} to {labels[-1]}; loan growth was {adv[years[-1]]:.0f}% and deposit growth "
           f"{dep[years[-1]]:.0f}% in {labels[-1]}.")
    merged_years = [fy(y) for y in years if adv[y] > cap]
    note = f"{', '.join(merged_years)} loan growth is merger-affected and not like-for-like." if merged_years else None
    return make_chart("growth_cycle", "main", "Loans, deposits and profit: yearly growth", option, alt,
                      {"columns": ["Year", "Loan growth %", "Deposit growth %", "Net profit growth %"], "rows": rows},
                      subtitle="Bars: loans (dark) and deposits (light). Line: net profit. Hatched bar: merger-affected year",
                      static_note=note, height=280)


def chart_credit_deposit(ctx: dict) -> Optional[dict]:
    years, vals = _years(ctx, "Credit-Deposit Ratio")
    if len(vals) < 3:
        return None
    labels = [fy(y) for y in years]
    data = [point(rnd(v, 1), tip=f"{lab}: credit-deposit ratio {v:.1f}%", label=f"{v:.0f}%", position="top", color=TEXT, bold=(i == len(vals) - 1))
            for i, (lab, v) in enumerate(zip(labels, vals))]
    line = {"name": "Credit-deposit ratio", "type": "line", "data": data, "symbolSize": 8, "lineStyle": {"color": ACCENT, "width": 3},
            "itemStyle": {"color": ACCENT},
            "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": GREY4, "type": "dashed", "width": 1.2},
                         "label": {"show": True, "formatter": "100%: every deposit lent", "position": "insideEndTop", "color": MUTED, "fontSize": 10},
                         "data": [{"yAxis": 100}]}}
    option = cc.base_option(grid={"left": 40, "right": 24, "top": 28, "bottom": 28}, xAxis=cc.cat_axis(labels),
                            yAxis=cc.val_axis(min=cc.nice_floor(min(vals) - 5, 10), max=cc.nice_ceil(max(max(vals), 100) + 5, 10), fmt="{value}%"), series=[line])
    alt = f"The credit-deposit ratio moved from {vals[0]:.0f}% in {labels[0]} to {vals[-1]:.0f}% in {labels[-1]}; 100% would mean every deposit is lent out."
    return make_chart("credit_deposit", "small", "How much of deposits is lent", option, alt,
                      {"columns": ["Year", "Credit-deposit ratio %"], "rows": [[l, f"{v:.1f}"] for l, v in zip(labels, vals)]},
                      subtitle="Line: the bank. Dashed: 100%", height=190)


# --------------------------------------------------------------------------------------
# panel 4: strength
# --------------------------------------------------------------------------------------
def chart_asset_quality(ctx: dict) -> Optional[dict]:
    g, n, c = (P.actual_series(ctx, x) for x in ("Gross NPA", "Net NPA", "Credit Cost"))
    years = [y for y in g if y in n]
    if len(years) < 3:
        return None
    labels = [fy(y) for y in years]
    meds = (ctx["peers"].get("bank") or {}).get("median_excl_subject") or {}
    gl = {"name": "Gross NPA", "type": "line", "symbolSize": 8, "lineStyle": {"color": ACCENT, "width": 3}, "itemStyle": {"color": ACCENT}, "z": 5,
          "data": [point(rnd(g[y], 2), tip=f"{lab}: gross NPA {pct(g[y])}", label=f"gross {pct(g[y])}", position="top", color=TEXT, bold=(i == len(years) - 1))
                   for i, (y, lab) in enumerate(zip(years, labels))]}
    if meds.get("gnpa"):
        gl["markLine"] = {"silent": True, "symbol": "none", "lineStyle": {"color": GREY4, "type": "dashed", "width": 1.2},
                          "label": {"show": True, "formatter": f"Peer median gross NPA {pct(meds['gnpa'])}", "position": "insideEndTop", "color": MUTED, "fontSize": 10},
                          "data": [{"yAxis": rnd(meds["gnpa"], 2)}]}
    nl = {"name": "Net NPA", "type": "line", "symbolSize": 7, "lineStyle": {"color": GREY5, "width": 2}, "itemStyle": {"color": GREY5},
          "data": [point(rnd(n[y], 2), tip=f"{lab}: net NPA {pct(n[y])}", label=pct(n[y]), position="bottom", color=MUTED) for y, lab in zip(years, labels)]}
    series = [gl, nl]
    if c:
        cy = [y for y in years if y in c]
        series.insert(0, {"name": "Credit cost", "type": "bar", "barWidth": 26, "itemStyle": {"color": GREY2}, "z": 1,
                          "data": [point(rnd(c[y], 2), tip=f"{fy(y)}: credit cost {pct(c[y])} of average loans") if y in c else None for y in years]})
    vmax = max(max(g[y], n[y], c.get(y, 0)) for y in years)
    option = cc.base_option(grid={"left": 40, "right": 16, "top": 28, "bottom": 36}, xAxis=cc.cat_axis(labels),
                            yAxis=cc.val_axis(min=0, max=cc.nice_ceil(vmax * 1.25, 0.5), fmt="{value}%"), series=series)
    rows = [[lab, pct(g[y]), pct(n[y]), pct(c.get(y))] for lab, y in zip(labels, years)]
    alt = (f"Gross NPA moved from {pct(g[years[0]])} in {labels[0]} to {pct(g[years[-1]])} in {labels[-1]} and net NPA from {pct(n[years[0]])} to {pct(n[years[-1]])}"
           + (f"; the peer median gross NPA is {pct(meds['gnpa'])}." if meds.get("gnpa") else "."))
    return make_chart("asset_quality", "main", "Loan quality: bad loans and what they cost", option, alt,
                      {"columns": ["Year", "Gross NPA", "Net NPA", "Credit cost"], "rows": rows},
                      subtitle="Lines: gross NPA (accent) and net NPA. Bars: credit cost. Dashed: peer median", height=280)


def _bullets(specs: list[dict], chart_id: str, title: str, subtitle: str, height: int) -> Optional[dict]:
    """One row per metric: own five-year range (bar), now (dot), peer median (diamond). Same pattern as the non-financial gauges."""
    if not specs:
        return None
    grids, xs, ys, series, titles, rows = [], [], [], [], [], []
    n = len(specs)
    for g, s in enumerate(specs):
        top = 6 + g * (100 // n)
        grids.append({"left": 24, "right": 32, "top": f"{top + 12}%", "height": "13%"})
        vmax = max(x for x in (s["hi"], s.get("peer"), s["now"]) if x is not None) * 1.2
        xs.append(cc.val_axis(grid_index=g, min=0, max=cc.nice_ceil(vmax, 1 if vmax > 5 else 0.5), fmt="{value}%", splitLine={"show": False}))
        ys.append(cc.cat_axis([""], grid_index=g, axisLine={"show": False}, axisLabel={"show": False}))
        titles.append({"text": f"{s['title']} ({s['dir']})", "left": 24, "top": f"{top}%", "textStyle": {"color": TEXT, "fontSize": 12, "fontWeight": "bold"}})
        series.append({"name": f"{s['title']} base", "type": "bar", "stack": f"r{g}", "xAxisIndex": g, "yAxisIndex": g, "data": [rnd(s["lo"], 3)],
                       "itemStyle": {"color": "transparent"}, "silent": True, "barWidth": 14, "tooltip": {"show": False}})
        series.append({"name": f"{s['title']}: own range", "type": "bar", "stack": f"r{g}", "xAxisIndex": g, "yAxisIndex": g, "barWidth": 14,
                       "itemStyle": {"color": GREY2},
                       "data": [point(rnd(s["hi"] - s["lo"], 3), tip=f"{s['title']}: own five-year range {pct(s['lo'])} to {pct(s['hi'])}",
                                      label=f"range {pct(s['lo'])} to {pct(s['hi'])}", position="insideRight", color=MUTED)]})
        if s.get("peer") is not None:
            series.append({"name": f"{s['title']}: peer median", "type": "scatter", "xAxisIndex": g, "yAxisIndex": g, "symbol": "diamond", "symbolSize": 14,
                           "itemStyle": {"color": GREY5},
                           "data": [point([rnd(s["peer"], 3), 0], tip=f"{s['title']}: peer median {pct(s['peer'])}", label=f"peer median {pct(s['peer'])}",
                                          position="bottom", color=MUTED)]})
        series.append({"name": f"{s['title']}: now", "type": "scatter", "xAxisIndex": g, "yAxisIndex": g, "symbolSize": 16,
                       "itemStyle": {"color": ACCENT, "borderColor": BG, "borderWidth": 2}, "z": 6,
                       "data": [point([rnd(s["now"], 3), 0], tip=f"{s['title']}: now {pct(s['now'])}", label=f"now {pct(s['now'])}", position="top", color=TEXT, bold=True)]})
        rows.append([s["title"], pct(s["now"]), f"{pct(s['lo'])} to {pct(s['hi'])}", pct(s.get("peer")) if s.get("peer") is not None else "n/a", s["dir"]])
    option = cc.base_option(title=titles, grid=grids, xAxis=xs, yAxis=ys, series=series)
    alt = "; ".join(f"{s['title']} is {pct(s['now'])} now against its own range of {pct(s['lo'])} to {pct(s['hi'])}"
                    + (f" and a peer median of {pct(s['peer'])}" if s.get("peer") is not None else "") for s in specs) + "."
    return make_chart(chart_id, "small", title, option, alt,
                      {"columns": ["Ratio", "Now", "Own five-year range", "Peer median", "Direction"], "rows": rows}, subtitle=subtitle, height=height)


def chart_capital_returns(ctx: dict) -> Optional[dict]:
    meds = (ctx["peers"].get("bank") or {}).get("median_excl_subject") or {}
    specs = []
    for title, name, peer_key, direction in (("CET1 capital ratio", "CET1 Ratio", "cet1", "higher is better"),
                                             ("Return on equity", "ROE", "roe", "higher is better"),
                                             ("Return on assets", "ROA", "roa", "higher is better")):
        s = P.actual_series(ctx, name)
        if len(s) >= 3:
            v = list(s.values())
            specs.append({"title": title, "now": v[-1], "lo": min(v), "hi": max(v), "peer": meds.get(peer_key), "dir": direction})
    return _bullets(specs, "capital_returns", "Capital and returns", "Dot: now. Bar: own five-year range. Diamond: peer median", 300)


# --------------------------------------------------------------------------------------
# panel 5: valuation
# --------------------------------------------------------------------------------------
def _strip(points: list[dict], subject: dict, faded: Optional[dict], median: Optional[float], median_label: str) -> list[dict]:
    peers = [point([rnd(p["value"], 2), 0], tip=p["tip"], label=f"{p['name']} {fmt_x(p['value'], 2)}", position="bottom", color=MUTED, offset=[0, 4 + (k % 2) * 16])
             for k, p in enumerate(sorted(points, key=lambda d: d["value"]))]
    series = [{"name": "Peers", "type": "scatter", "symbolSize": 12, "itemStyle": {"color": GREY3}, "data": peers}]
    if median is not None:
        series[0]["markLine"] = {"silent": True, "symbol": "none", "lineStyle": {"color": GREY4, "type": "dashed", "width": 1.2},
                                 "label": {"show": True, "formatter": median_label, "position": "end", "color": MUTED, "fontSize": 10},
                                 "data": [{"xAxis": rnd(median, 2)}]}
    if faded:
        series.append({"name": faded["name"], "type": "scatter", "symbolSize": 20, "itemStyle": {"color": ACCENT, "opacity": 0.3, "borderColor": ACCENT, "borderWidth": 1},
                       "data": [point([rnd(faded["value"], 2), 0], tip=faded["tip"], label=faded["label"], position="bottom", color=MUTED, offset=[0, 24])]})
    series.append({"name": subject["name"], "type": "scatter", "symbolSize": 24, "itemStyle": {"color": ACCENT, "borderColor": BG, "borderWidth": 2}, "z": 6,
                   "data": [point([rnd(subject["value"], 2), 0], tip=subject["tip"], label=subject["label"], position="top", color=TEXT, bold=True)]})
    return series


def chart_pb_peer_strip(ctx: dict) -> Optional[dict]:
    pb = (ctx["peers"].get("bank") or {})
    rows_ = pb.get("rows") or {}
    sub = _tag(ctx)
    now_sub = ctx["current_valuation"].get("pb_now")
    avg_sub = (rows_.get(sub) or {}).get("pb_fy_avg")
    if not rows_ or not now_sub or not avg_sub:
        return None
    meds = pb.get("median_excl_subject") or {}
    price = ctx["market"]["price"]
    latest = ctx["meta"]["latest_actual_year"]
    avg_price = (ctx["fy_series"].get(latest) or {}).get("avg_price")
    all_v = [v for r in rows_.values() for v in (r.get("pb_now"), r.get("pb_fy_avg")) if v] + [now_sub, avg_sub]
    axis_max = cc.nice_ceil(max(all_v) * 1.15, 0.5)

    def variant(key: str) -> dict:
        if key == "today":
            pts = [{"name": t, "value": r["pb_now"], "tip": f"{t}: P/B {fmt_x(r['pb_now'], 2)} at today's price (Rs {r['price_now']:,.2f})"}
                   for t, r in rows_.items() if t != sub and r.get("pb_now")]
            subj = {"name": sub, "value": now_sub, "tip": f"{sub}: P/B {fmt_x(now_sub, 2)} at today's price (Rs {price:,.2f})", "label": f"{sub} {fmt_x(now_sub, 2)} today"}
            faded = {"name": "Old report multiple", "value": avg_sub, "tip": f"{sub}: P/B {fmt_x(avg_sub, 2)} at the fiscal-year average price (the multiple the old report used)",
                     "label": f"old report {fmt_x(avg_sub, 2)}"}
            med, axis = meds.get("pb_now"), "P/B, x (today's prices, latest book value)"
        else:
            pts = [{"name": t, "value": r["pb_fy_avg"], "tip": f"{t}: P/B {fmt_x(r['pb_fy_avg'], 2)} at its fiscal-year average price"}
                   for t, r in rows_.items() if t != sub and r.get("pb_fy_avg")]
            subj = {"name": sub, "value": avg_sub, "tip": f"{sub}: P/B {fmt_x(avg_sub, 2)} at the fiscal-year average price", "label": f"{sub} {fmt_x(avg_sub, 2)} at report price"}
            faded = {"name": "Today", "value": now_sub, "tip": f"{sub}: P/B {fmt_x(now_sub, 2)} at today's price", "label": f"today {fmt_x(now_sub, 2)}"}
            med, axis = meds.get("pb_fy_avg"), f"P/B, x ({pb['year']}, fiscal-year average prices)"
        return cc.base_option(grid={"left": 24, "right": 36, "top": 64, "bottom": 40},
                              xAxis=cc.val_axis(min=0, max=axis_max, fmt="{value}x", name=axis, nameLocation="middle", nameGap=26, nameTextStyle={"color": MUTED, "fontSize": 11}),
                              yAxis=cc.cat_axis([""], axisLine={"show": False}, axisLabel={"show": False}),
                              series=_strip(pts, subj, faded, med, f"Peer median {fmt_x(med, 2)}" if med else ""))
    rows = []
    for t in sorted(rows_, key=lambda t: (t != sub, t)):
        r = rows_[t]
        rows.append([t + (" (subject)" if t == sub else ""), fmt_x(r.get("pb_now"), 2), fmt_x(r.get("pb_fy_avg"), 2)])
    rows.append(["Peer median (excl. subject)", fmt_x(meds.get("pb_now"), 2), fmt_x(meds.get("pb_fy_avg"), 2)])
    note = f"At the report's earlier price of about Rs {avg_price:.0f}, P/B was {fmt_x(avg_sub, 2)}." if avg_price else None
    alt = (f"At today's price of {fmt_rs(price)} {sub} trades at {fmt_x(now_sub, 2)} book value against a peer median of {fmt_x(meds.get('pb_now'), 2)}, "
           f"and {fmt_x(avg_sub, 2)} at its fiscal-year average price.")
    return make_chart("pb_peer_strip", "main", "Where the price-to-book sits against peers", variant("today"), alt,
                      {"columns": ["Bank", "P/B at today's price", "P/B at fiscal-year average price"], "rows": rows},
                      subtitle="P/B for each bank; the dashed line is the peer median",
                      variants=[{"key": "today", "label": "At today's price", "option": variant("today")},
                                {"key": "report", "label": "At the report's price", "option": variant("report")}],
                      default_variant="today", static_note=note, height=250)


def chart_fair_value_ruler(ctx: dict, cases: list[dict], price: Optional[float]) -> Optional[dict]:
    if not cases or not price:
        return None
    labels = [c["label"] for c in cases]
    vmax = max([c["high_price"] for c in cases] + [price]) * 1.12
    vmin = min([c["low_price"] for c in cases] + [price]) * 0.9
    base = [{"value": rnd(c["low_price"], 0), "tip": f"{c['label']}: from {fmt_rs(c['low_price'], 0)}"} for c in cases]
    span = [point(rnd(c["high_price"] - c["low_price"], 0),
                  tip=f"{c['label']}: {fmt_rs(c['low_price'], 0)} to {fmt_rs(c['high_price'], 0)} per share ({c['basis']})",
                  label=f"{fmt_rs(c['low_price'], 0)} to {fmt_rs(c['high_price'], 0)}", position="right", color=TEXT) for c in cases]
    series = [{"name": "base", "type": "bar", "stack": "fv", "data": base, "itemStyle": {"color": "transparent"}, "silent": True, "barWidth": 16, "tooltip": {"show": False}},
              {"name": "Fair-value range", "type": "bar", "stack": "fv", "data": span, "barWidth": 16, "itemStyle": {"color": GREY3},
               "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": ACCENT, "width": 2},
                            "label": {"show": True, "formatter": f"Today {fmt_rs(price, 0)}", "position": "end", "color": TEXT, "fontSize": 11},
                            "data": [{"xAxis": rnd(price, 2)}]}}]
    option = cc.base_option(grid={"left": 190, "right": 110, "top": 28, "bottom": 28},
                            xAxis=cc.val_axis(min=cc.nice_floor(vmin, 100), max=cc.nice_ceil(vmax, 100), fmt="Rs {value}"),
                            yAxis=cc.cat_axis(labels, inverse=True, axisLabel={"color": TEXT, "fontSize": 11, "width": 170, "overflow": "break"}), series=series)
    rows = [[c["label"], fmt_rs(c["low_price"], 0), fmt_rs(c["high_price"], 0), f"{c['vs_price_pct_low']:+.0f}% to {c['vs_price_pct_high']:+.0f}%", c["basis"]] for c in cases]
    above = sum(1 for c in cases if price > c["high_price"])
    below = sum(1 for c in cases if price < c["low_price"])
    alt = (f"Today's price of {fmt_rs(price, 0)} sits above {above} of {len(cases)} fair-value ranges and below {below}; the ranges run from "
           f"{fmt_rs(min(c['low_price'] for c in cases), 0)} to {fmt_rs(max(c['high_price'] for c in cases), 0)} per share.")
    return make_chart("fair_value_ruler", "small", "Price against fair-value ranges", option, alt,
                      {"columns": ["Case", "Low per share", "High per share", "Against today's price", "Basis"], "rows": rows},
                      subtitle="Bars: fair value per share from each case. Line: today's price", height=250)


def chart_pb_vs_roe(ctx: dict) -> Optional[dict]:
    pb = (ctx["peers"].get("bank") or {})
    rows_ = pb.get("rows") or {}
    sub = _tag(ctx)
    pts = [(t, r["roe"], r.get("pb_now")) for t, r in rows_.items() if r.get("roe") is not None and r.get("pb_now")]
    if len(pts) < 4:
        return None
    others = [(t, x, y) for t, x, y in pts if t != sub]
    mine = next(((x, y) for t, x, y in pts if t == sub), None)
    if not mine or len(others) < 3:
        return None
    xs, ys = [x for _, x, _ in others], [y for _, _, y in others]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    icpt = my - slope * mx
    x0, x1 = min(xs + [mine[0]]) - 0.5, max(xs + [mine[0]]) + 0.5
    fit = icpt + slope * mine[0]
    series = [{"name": "Peers", "type": "scatter", "symbolSize": 12, "itemStyle": {"color": GREY3},
               "data": [point([rnd(x, 2), rnd(y, 2)], tip=f"{t}: ROE {pct(x, 1)}, P/B {fmt_x(y, 2)} at today's price", label=t, position="right", color=MUTED) for t, x, y in others]},
              {"name": "Rough peer trend", "type": "line", "showSymbol": False, "silent": True, "lineStyle": {"color": GREY4, "type": "dashed", "width": 1.2},
               "data": [[rnd(x0, 2), rnd(icpt + slope * x0, 2)], [rnd(x1, 2), rnd(icpt + slope * x1, 2)]]},
              {"name": sub, "type": "scatter", "symbolSize": 20, "itemStyle": {"color": ACCENT, "borderColor": BG, "borderWidth": 2}, "z": 6,
               "data": [point([rnd(mine[0], 2), rnd(mine[1], 2)], tip=f"{sub}: ROE {pct(mine[0], 1)}, P/B {fmt_x(mine[1], 2)} at today's price",
                              label=f"{sub} ROE {pct(mine[0], 1)}, P/B {fmt_x(mine[1], 2)}", position="top", color=TEXT, bold=True)]}]
    option = cc.base_option(grid={"left": 48, "right": 30, "top": 28, "bottom": 44},
                            xAxis=cc.val_axis(min=cc.nice_floor(x0, 1), max=cc.nice_ceil(x1, 1), fmt="{value}%", name="Return on equity, latest year", nameLocation="middle", nameGap=28),
                            yAxis=cc.val_axis(min=0, max=cc.nice_ceil(max(ys + [mine[1]]) * 1.2, 0.5), fmt="{value}x", name="P/B at today's price"), series=series)
    rel = "above" if mine[1] > fit * 1.1 else ("below" if mine[1] < fit * 0.9 else "close to")
    rows = [[t + (" (subject)" if t == sub else ""), pct(x, 1), fmt_x(y, 2)] for t, x, y in sorted(pts, key=lambda p: (p[0] != sub, p[0]))]
    alt = (f"{sub} earns a {pct(mine[0], 1)} return on equity and trades at {fmt_x(mine[1], 2)} book; a rough trend through its peers would put a bank with that return "
           f"at about {fmt_x(fit, 2)}, so it trades {rel} what its return suggests.")
    return make_chart("pb_vs_roe", "small", "Does the return justify the price-to-book?", option, alt,
                      {"columns": ["Bank", "ROE, latest year", "P/B, today"], "rows": rows},
                      subtitle="Each dot is a bank. Dashed: rough trend through the peers", height=250)


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def _try(name: str, fn, out: list) -> None:
    try:
        chart = fn()
    except Exception as e:
        LAST_ERRORS.append(f"{name}: {type(e).__name__}: {e}")
        return
    if chart:
        out.append(chart)
    else:
        LAST_ERRORS.append(f"{name}: skipped (missing input)")


def build_bank_panel_charts(ctx: dict, analysis_dir: str, variant: Optional[dict] = None, cases: Optional[list] = None,
                            price: Optional[float] = None) -> dict[str, list[dict]]:
    LAST_ERRORS.clear()
    variant = variant or P.load_variant()
    panels: dict[str, list[dict]] = {pid: [] for pid in PANEL_IDS}
    prep = tf = None
    try:
        prep = nf._prepared_prices(analysis_dir)
        if prep is not None:
            from modules import technical_analysis as ta
            pve = ((ctx.get("extras") or {}).get("technical") or {}).get("price_vs_earnings") or {}
            info = {"eps_fy_now": pve.get("eps_fy_now"), "eps_fy_prev": pve.get("eps_fy_prev"), "eps_ttm": pve.get("eps_ttm")}
            tf = ta.compute_technical_facts(prep[["date", "open", "high", "low", "close", "volume"]], info)
    except Exception as e:
        LAST_ERRORS.append(f"price: {type(e).__name__}: {e}")
    builders = {
        "income_split": lambda: chart_income_split(ctx), "nim_trend": lambda: chart_nim_trend(ctx),
        "growth_cycle": lambda: chart_growth_cycle(ctx), "credit_deposit": lambda: chart_credit_deposit(ctx),
        "asset_quality": lambda: chart_asset_quality(ctx), "capital_returns": lambda: chart_capital_returns(ctx),
        "pb_peer_strip": lambda: chart_pb_peer_strip(ctx), "fair_value_ruler": lambda: chart_fair_value_ruler(ctx, cases or [], price),
        "pb_vs_roe": lambda: chart_pb_vs_roe(ctx), "price_vs_earnings": lambda: nf.chart_price_vs_earnings(ctx),
        "price_trend": lambda: nf.chart_price_trend(ctx, prep, tf) if prep is not None and tf and tf.get("available", True) else None,
        "momentum": lambda: nf.chart_momentum(ctx, prep, tf) if prep is not None and tf and tf.get("available", True) else None,
    }
    for pid, spec in (variant.get("panels") or {}).items():
        for cid in spec.get("charts") or []:
            _try(cid, builders[cid], panels[pid])
    return panels
