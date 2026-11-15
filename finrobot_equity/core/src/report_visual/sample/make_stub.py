#!/usr/bin/env python
# coding: utf-8
"""Build sample/stub.report.json: a hand-made, schema-valid model used to develop and test the renderers.

Figures follow the WELSPUNLIV reference case but are fixed literals here (no data layer needed).
Run: python make_stub.py   (writes stub.report.json next to this file)
"""
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
YEARS = ["FY22", "FY23", "FY24", "FY25", "FY26", "2027E", "2028E", "2029E"]


def tip(value, text, label=None):
    d = {"value": value, "tip": text}
    if label:
        d["label"] = {"show": True, "position": "top", "formatter": label}
    return d


def chip(metric, label, value, text, unit, period, verdict, yard, scale, so_what, source="report_data", calc=""):
    yt = {"own_history": "vs own history", "peers": "vs peers", "general_norm": "General norm, not company-specific"}[yard]
    return {"id": metric, "metric": metric, "label": label, "value": value, "value_text": text, "unit": unit, "period": period,
            "verdict": verdict, "icon": {"good": "check", "neutral": "dash", "concern": "alert"}[verdict],
            "yardstick": yard, "yardstick_text": yt, "scale": scale, "so_what": so_what, "source": source, "calc": calc}


def table(cols, rows):
    return {"columns": cols, "rows": [[str(c) for c in r] for r in rows]}


# ----------------------------------------------------------------------------- panel 1
p1_chart = {
    "id": "price_vs_earnings", "role": "main", "title": "Price against earnings", "subtitle": "Change over the last year, %",
    "option": {
        "grid": {"left": 150, "right": 60, "top": 10, "bottom": 24},
        "xAxis": {"type": "value", "axisLine": {"show": False}, "splitLine": {"lineStyle": {"color": "@grid"}}, "axisLabel": {"color": "@muted"}},
        "yAxis": {"type": "category", "inverse": True, "data": ["Share price, 1 year", "EPS, last full year", "EPS, TTM vs last year"],
                  "axisLine": {"lineStyle": {"color": "@grey4"}}, "axisLabel": {"color": "@text"}},
        "series": [{"type": "bar", "barWidth": 22, "data": [
            {"value": 82, "tip": "Share price: +82% over one year", "itemStyle": {"color": "@accent"}, "label": {"show": True, "position": "right", "formatter": "+82%"}},
            {"value": -68, "tip": "EPS FY26 against FY25: -68%", "itemStyle": {"color": "@grey3"}, "label": {"show": True, "position": "left", "formatter": "-68%"}},
            {"value": 115, "tip": "TTM EPS against FY26 EPS: +115%", "itemStyle": {"color": "@grey3"}, "label": {"show": True, "position": "right", "formatter": "+115%"}}]}]},
    "alt": "Bar chart. The share price rose 82% in a year. EPS fell 68% in the last full year but TTM EPS is 115% above that low base.",
    "datatable": table(["Measure", "Change %"], [["Share price, 1 year", "+82"], ["EPS, last full year", "-68"], ["EPS, TTM vs last year", "+115"]]),
    "variants": None, "default_variant": None, "static_note": None, "height": 190}

summary = {
    "id": "summary", "n": 1, "nav": "Summary", "question": "Should I worry, and about what?",
    "headline": "Sound product, hurt by costs, priced ahead",
    "charts": [p1_chart], "chips": [],
    "extra": {
        "lights": [
            {"key": "business", "label": "Business health", "light": "neutral", "reason": "Pricing holds, but interest cover is thin and sales are shrinking."},
            {"key": "profit", "label": "Profit trend", "light": "concern", "reason": "EBITDA margin is near its five-year low; TTM shows a recovery."},
            {"key": "price", "label": "Price vs value", "light": "concern", "reason": "Fair value per share is below today's price in every case."},
            {"key": "momentum", "label": "Momentum", "light": "neutral", "reason": "Uptrend intact, but late, stretched and thinly traded."}],
        "price_vs_earnings": {"price_change_1y_pct": 82.3, "eps_fy_change_pct": -68.0, "eps_ttm_change_pct": 114.7,
                              "sentence": "The share price is behind TTM earnings but far ahead of last full-year earnings."},
        "header_strip": {}},
    "notes": []}

# ----------------------------------------------------------------------------- panel 2
cm = [45.3, 44.2, 47.2, 46.2, 46.2, 47.2, 48.2, 49.2]
sg = [30.7, 34.9, 33.1, 33.9, 37.8, 37.3, 36.8, 36.3]
gap = [round(a - b, 1) for a, b in zip(cm, sg)]
na = [None] * 4
profit_main = {
    "id": "margin_scissors", "role": "main", "title": "Prices held, overheads rose", "subtitle": "Contribution margin and SG&A as % of revenue; forecasts dashed",
    "option": {
        "grid": {"left": 44, "right": 150, "top": 14, "bottom": 28},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": YEARS, "axisLine": {"lineStyle": {"color": "@grey4"}}, "axisLabel": {"color": "@muted"}},
        "yAxis": {"type": "value", "min": 25, "max": 55, "axisLabel": {"color": "@muted", "formatter": "{value}%"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
        "series": [
            {"name": "SG&A base", "type": "line", "stack": "gap", "symbol": "none", "lineStyle": {"opacity": 0}, "areaStyle": {"opacity": 0}, "data": sg, "silent": True},
            {"name": "Gap", "type": "line", "stack": "gap", "symbol": "none", "lineStyle": {"opacity": 0}, "areaStyle": {"color": "@grey5", "opacity": 0.7}, "data": gap, "silent": True},
            {"name": "Contribution margin", "type": "line", "symbol": "circle", "symbolSize": 6, "lineStyle": {"color": "@accent", "width": 3}, "itemStyle": {"color": "@accent"},
             "data": [tip(v, f"{y}: contribution margin {v}%") for y, v in zip(YEARS[:5], cm[:5])] + [None] * 3,
             "endLabel": {"show": False}},
            {"name": "Contribution margin (forecast)", "type": "line", "symbol": "circle", "symbolSize": 5, "lineStyle": {"color": "@accent", "width": 2, "type": "dashed"}, "itemStyle": {"color": "@accent"},
             "data": [None] * 4 + [tip(cm[4], "FY26: contribution margin 46.2%")] + [tip(v, f"{y}: contribution margin {v}% (forecast)") for y, v in zip(YEARS[5:], cm[5:])],
             "endLabel": {"show": True, "formatter": "Contribution margin", "color": "@accent", "distance": 6}},
            {"name": "SG&A as % of revenue", "type": "line", "symbol": "circle", "symbolSize": 6, "lineStyle": {"color": "@grey1", "width": 3}, "itemStyle": {"color": "@grey1"},
             "data": [tip(v, f"{y}: SG&A {v}% of revenue") for y, v in zip(YEARS[:5], sg[:5])] + [None] * 3,
             "markPoint": {"symbol": "pin", "symbolSize": 44, "itemStyle": {"color": "@grey1"}, "label": {"color": "@bg", "formatter": "37.8%", "fontSize": 10},
                           "data": [{"coord": ["FY26", 37.8], "tip": "SG&A 37.8%, five-year high"}]}},
            {"name": "SG&A (forecast)", "type": "line", "symbol": "circle", "symbolSize": 5, "lineStyle": {"color": "@grey1", "width": 2, "type": "dashed"}, "itemStyle": {"color": "@grey1"},
             "data": [None] * 4 + [tip(sg[4], "FY26: SG&A 37.8% of revenue")] + [tip(v, f"{y}: SG&A {v}% of revenue (forecast)") for y, v in zip(YEARS[5:], sg[5:])],
             "endLabel": {"show": True, "formatter": "SG&A % of revenue", "color": "@grey1", "distance": 6},
             "markArea": {"silent": True, "itemStyle": {"color": "@grey5", "opacity": 0.25}, "label": {"show": True, "position": "insideTop", "formatter": "Forecast", "color": "@muted"},
                          "data": [[{"xAxis": "2027E"}, {"xAxis": "2029E"}]]}}]},
    "alt": "Line chart. SG&A rose from 30.7% to 37.8% of revenue between FY22 and FY26 while contribution margin stayed near 46%. The shaded gap between the lines narrowed. Forecasts are dashed.",
    "datatable": table(["Year", "Contribution margin %", "SG&A % of revenue"], [[y, a, b] for y, a, b in zip(YEARS, cm, sg)]),
    "variants": None, "default_variant": None, "static_note": None, "height": 300}
profit_small = {
    "id": "waterfall", "role": "small", "title": "From revenue to EBITDA, FY26", "subtitle": "Rs crore",
    "option": {
        "grid": {"left": 12, "right": 12, "top": 24, "bottom": 28, "containLabel": True},
        "xAxis": {"type": "category", "data": ["Revenue", "Direct costs", "Contribution", "SG&A", "EBITDA"], "axisLabel": {"color": "@muted", "interval": 0, "fontSize": 10}},
        "yAxis": {"type": "value", "show": False},
        "series": [
            {"type": "bar", "stack": "w", "itemStyle": {"color": "transparent"}, "silent": True, "data": [0, 5346, 0, 793, 0]},
            {"type": "bar", "stack": "w", "barWidth": 34, "data": [
                {"value": 9399, "tip": "Revenue Rs 9,399 cr", "itemStyle": {"color": "@grey3"}, "label": {"show": True, "position": "top", "formatter": "9,399"}},
                {"value": 5053, "tip": "Direct costs Rs 5,053 cr", "itemStyle": {"color": "@grey4"}, "label": {"show": True, "position": "top", "formatter": "-5,053"}},
                {"value": 4346, "tip": "Contribution Rs 4,346 cr", "itemStyle": {"color": "@accent"}, "label": {"show": True, "position": "top", "formatter": "4,346"}},
                {"value": 3553, "tip": "SG&A Rs 3,553 cr", "itemStyle": {"color": "@grey1"}, "label": {"show": True, "position": "top", "formatter": "-3,553"}},
                {"value": 793, "tip": "EBITDA Rs 793 cr", "itemStyle": {"color": "@accent"}, "label": {"show": True, "position": "top", "formatter": "793"}}]}]},
    "alt": "Waterfall chart. Revenue of Rs 9,399 crore less Rs 5,053 crore of direct costs leaves Rs 4,346 crore of contribution. SG&A of Rs 3,553 crore leaves EBITDA of Rs 793 crore.",
    "datatable": table(["Step", "Rs crore"], [["Revenue", "9,399"], ["Direct costs", "-5,053"], ["Contribution", "4,346"], ["SG&A", "-3,553"], ["EBITDA", "793"]]),
    "variants": None, "default_variant": None, "static_note": None, "height": 240}
profit = {
    "id": "profit_engine", "n": 2, "nav": "Profit engine", "question": "Where did the profit go?",
    "headline": "Prices held. Costs broke the margin.",
    "charts": [profit_main, profit_small],
    "chips": [
        chip("contribution_margin", "Contribution margin", 46.2, "46.2%", "%", "FY26", "good", "own_history",
             {"min": 44.2, "max": 47.2, "peer_median": None, "higher_is_better": True, "value": 46.2},
             "Revenue swung 33 points but this margin moved 3. Pricing is holding; the profit problem lies in costs below this line.",
             calc="Contribution profit / revenue, FY22 to FY26. Verdict: top third of own five-year range."),
        chip("sga_ratio", "SG&A as % of revenue", 37.8, "37.8%", "%", "FY26", "concern", "own_history",
             {"min": 30.7, "max": 37.8, "peer_median": 35.7, "higher_is_better": False, "value": 37.8},
             "Overheads take more of each sale than at any time in five years, which is what squeezes profit when sales fall.",
             calc="SG&A / revenue. Lower is better. Peer median from the peer set."),
        chip("operating_leverage", "Operating leverage", 44.1, "Rs 44 per Rs 100", "", "FY25 to FY26", "concern", "own_history",
             {"min": 0, "max": 100, "peer_median": None, "higher_is_better": False, "value": 44.1},
             "A large share of each lost rupee of sales comes straight out of profit, so earnings swing more than sales.",
             source="calculated", calc="Change in EBITDA / change in revenue x 100, FY25 to FY26.")],
    "extra": {"bridge": {"from": "FY22", "to": "FY26", "d_ebitda_margin_pts": -6.2, "d_contribution_margin_pts": 0.9, "d_sga_pct_pts": 7.1,
                         "text": "EBITDA margin moved -6.2 points from FY22 to FY26: contribution margin +0.9 points, SG&A as % of revenue +7.1 points."},
              "operating_leverage": {"ebitda_per_100_revenue": 44.1, "text": "About Rs 44 of every Rs 100 of lost sales came out of EBITDA from FY25 to FY26."}},
    "notes": []}

# ----------------------------------------------------------------------------- panel 3
growth = [-13.1, 19.6, 8.9, -10.9]
ebitda = [1359, 753, 1369, 1298, 793]
cyc_main = {
    "id": "growth_vs_ebitda", "role": "main", "title": "Sales swings drive earnings swings", "subtitle": "Revenue growth (bars, left axis) and EBITDA (line, right axis)",
    "option": {
        "grid": {"left": 50, "right": 56, "top": 44, "bottom": 28},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": YEARS[:5], "axisLabel": {"color": "@muted"}, "axisLine": {"lineStyle": {"color": "@grey4"}}},
        "yAxis": [{"type": "value", "name": "Revenue growth, %", "nameTextStyle": {"color": "@muted"}, "axisLabel": {"color": "@muted"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
                  {"type": "value", "name": "EBITDA, Rs cr", "nameTextStyle": {"color": "@muted"}, "min": 0, "max": 1800, "axisLabel": {"color": "@muted"}, "splitLine": {"show": False}}],
        "series": [
            {"name": "Revenue growth", "type": "bar", "barWidth": 30, "data": [None] + [
                {"value": v, "tip": f"{y}: revenue growth {v}%", "itemStyle": {"color": "@concern" if v < 0 else "@grey3"}, "label": {"show": True, "position": "top" if v > 0 else "bottom", "formatter": f"{v}%"}}
                for y, v in zip(YEARS[1:5], growth)]},
            {"name": "EBITDA", "type": "line", "yAxisIndex": 1, "symbol": "circle", "symbolSize": 7, "lineStyle": {"color": "@accent", "width": 3}, "itemStyle": {"color": "@accent"},
             "data": [tip(v, f"{y}: EBITDA Rs {v:,} cr", f"{v:,}") for y, v in zip(YEARS[:5], ebitda)],
             "markArea": {"silent": True, "itemStyle": {"color": "@grey5", "opacity": 0.45}, "label": {"show": True, "formatter": "Steadiest peer: Rs 1,220 to 1,280 cr", "position": "insideTop", "color": "@muted", "fontSize": 10},
                          "data": [[{"yAxis": 1220}, {"yAxis": 1280}]]}}]},
    "alt": "Combined chart. Revenue fell 13.1% in FY23 and 10.9% in FY26, and EBITDA fell with it from Rs 1,359 crore to Rs 753 crore and from Rs 1,298 crore to Rs 793 crore. The steadiest peer stayed within Rs 1,220 to 1,280 crore.",
    "datatable": table(["Year", "Revenue growth %", "EBITDA Rs cr"], [[y, g, e] for y, g, e in zip(YEARS[:5], ["n/a"] + growth, ebitda)]),
    "variants": None, "default_variant": None, "static_note": None, "height": 300}
cyc = {
    "id": "cyclicality", "n": 3, "nav": "Cyclicality", "question": "Is this a one-off dip?",
    "headline": "Second downturn in four years, same pattern.",
    "charts": [cyc_main],
    "chips": [
        chip("ebitda_margin", "EBITDA margin", 8.4, "8.4%", "%", "FY26", "concern", "own_history",
             {"min": 8.4, "max": 14.6, "peer_median": 13.0, "higher_is_better": True, "value": 8.4},
             "Operating profit per rupee of sales is at its low point, so each further fall in sales hurts earnings sharply.",
             calc="EBITDA / revenue, FY22 to FY26."),
        chip("earnings_volatility", "Earnings volatility", 6.2, "6.2 pts", "pts", "FY22 to FY26", "concern", "peers",
             {"min": 1.0, "max": 8.0, "peer_median": 2.1, "higher_is_better": False, "value": 6.2},
             "Profit margins swing far more than at the steadiest peers, which makes the earnings path harder to trust.",
             source="calculated", calc="Range of EBITDA margin over five years, compared with the peer median range.")],
    "extra": {}, "notes": []}

# ----------------------------------------------------------------------------- panel 4
cfo = [587, 756, 533, 688, 1175, 1910]
conv = ["43%", "100%", "39%", "53%", "148%", "179%"]
yrs_t = YEARS[:5] + ["TTM"]
eb_t = ebitda + [1068]
strength_main = {
    "id": "ebitda_vs_cfo", "role": "main", "title": "Does profit turn into cash?", "subtitle": "EBITDA and operating cash flow, Rs crore; conversion % above each pair",
    "option": {
        "grid": {"left": 50, "right": 16, "top": 36, "bottom": 30},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": yrs_t, "axisLabel": {"color": "@muted"}, "axisLine": {"lineStyle": {"color": "@grey4"}}},
        "yAxis": {"type": "value", "min": 0, "axisLabel": {"color": "@muted"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
        "series": [
            {"name": "EBITDA", "type": "bar", "barGap": "8%", "itemStyle": {"color": "@grey3"}, "data": [tip(v, f"{y}: EBITDA Rs {v:,} cr") for y, v in zip(yrs_t, eb_t)]},
            {"name": "Operating cash flow", "type": "bar", "itemStyle": {"color": "@accent"},
             "data": [tip(v, f"{y}: operating cash flow Rs {v:,} cr, {c} of EBITDA", c) for y, v, c in zip(yrs_t, cfo, conv)]}]},
    "alt": "Paired bar chart of EBITDA and operating cash flow. Cash conversion was 43%, 100%, 39%, 53% and 148% from FY22 to FY26. It rose when sales fell, in FY23 and FY26.",
    "datatable": table(["Year", "EBITDA Rs cr", "Operating cash flow Rs cr", "Conversion"], [[y, e, c, k] for y, e, c, k in zip(yrs_t, eb_t, cfo, conv)]),
    "variants": None, "default_variant": None, "static_note": None, "height": 290}


def bullet(cid, title, sub, cats, values, texts, good_dir):
    return {"id": cid, "role": "small", "title": title, "subtitle": sub,
            "option": {"grid": {"left": 12, "right": 20, "top": 10, "bottom": 24, "containLabel": True},
                       "xAxis": {"type": "category", "data": cats, "axisLabel": {"color": "@muted"}},
                       "yAxis": {"type": "value", "min": 0, "axisLabel": {"color": "@muted"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
                       "series": [{"type": "bar", "barWidth": 30, "data": [
                           {"value": v, "tip": t, "itemStyle": {"color": "@accent" if i == len(values) - 1 else "@grey3"}, "label": {"show": True, "position": "top", "formatter": f"{v}x"}}
                           for i, (v, t) in enumerate(zip(values, texts))]}]},
            "alt": f"{title}: {cats[0]} {values[0]}x, {cats[-1]} {values[-1]}x. {good_dir}",
            "datatable": table(["Year", title], [[c, v] for c, v in zip(cats, values)]),
            "variants": None, "default_variant": None, "static_note": None, "height": 200}


strength = {
    "id": "strength", "n": 4, "nav": "Strength", "question": "Can it fund itself and carry its debt?",
    "headline": "Debt falling; cash strong, partly one-off.",
    "charts": [strength_main,
               bullet("debt_equity", "Debt to equity", "Lower is better", ["FY22", "FY26"], [0.8, 0.37], ["FY22: debt to equity 0.80x", "FY26: debt to equity 0.37x"], "Improving."),
               bullet("interest_cover", "Interest coverage", "Higher is better", ["FY22", "FY26"], [7.7, 2.8], ["FY22: interest cover 7.7x", "FY26: interest cover 2.8x"], "Weakening.")],
    "chips": [
        chip("cash_conversion", "Cash conversion", 148, "148%", "%", "FY26", "concern", "own_history",
             {"min": 39, "max": 148, "peer_median": None, "higher_is_better": True, "value": 148},
             "Cash comes in when sales fall and goes out when they rise, so FY26 cash is flattered.",
             source="calculated", calc="Operating cash flow / EBITDA each year; correlation with revenue growth -0.89."),
        chip("debt_equity", "Debt to equity", 0.37, "0.37x", "x", "FY26", "good", "own_history",
             {"min": 0.37, "max": 0.8, "peer_median": 0.18, "higher_is_better": False, "value": 0.37},
             "Borrowing has fallen steadily, which gives room to ride out a weak year.", calc="Total debt / shareholders' equity, FY22 to FY26."),
        chip("interest_coverage", "Interest coverage", 2.8, "2.8x", "x", "FY26", "concern", "general_norm",
             {"min": 2.0, "max": 4.0, "peer_median": None, "higher_is_better": True, "value": 2.8},
             "Profit covers interest only thinly, so a further earnings dip would strain debt service.", source="general_norm",
             calc="EBIT / interest expense. General norm: below 3x is thin.")],
    "extra": {},
    "notes": ["Capex is reported separately from other investing cash flow: capex was Rs 799 cr in FY26 and free cash flow Rs 376 cr."]}

# ----------------------------------------------------------------------------- panel 5
peers = [("VTL", 11.4), ("ICIL", 16.0), ("TRIDENT", 17.8), ("GARFIBRES", 25.7), ("KPRMILL", 26.9)]


def strip_option(subject_x, subject_label, faded=None):
    data = [{"value": [v, 0], "name": n, "tip": f"{n}: EV/EBITDA {v}x", "label": {"show": True, "position": "top", "formatter": n, "color": "@muted", "fontSize": 10}, "itemStyle": {"color": "@grey3"}} for n, v in peers]
    data.append({"value": [subject_x, 0], "name": "Welspun", "tip": subject_label, "symbolSize": 20,
                 "label": {"show": True, "position": "bottom", "formatter": f"Welspun {subject_x}x", "color": "@accent", "fontWeight": "bold"}, "itemStyle": {"color": "@accent"}})
    series = [{"type": "scatter", "symbolSize": 14, "data": data,
               "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": "@grey2", "type": "dashed"}, "label": {"formatter": "Peer median 17.8x", "color": "@muted", "position": "end"}, "data": [{"xAxis": 17.8}]}}]
    if faded:
        series.append({"type": "scatter", "symbolSize": 14, "data": [{"value": [faded, 0], "tip": f"At the report's price of about Rs 129: {faded}x", "itemStyle": {"color": "@accent", "opacity": 0.3}, "label": {"show": True, "position": "bottom", "formatter": f"{faded}x (report price)", "color": "@muted", "fontSize": 10}}]})
    return {"grid": {"left": 20, "right": 40, "top": 40, "bottom": 50},
            "xAxis": {"type": "value", "min": 0, "max": 35, "name": "EV/EBITDA, x", "nameLocation": "middle", "nameGap": 28, "nameTextStyle": {"color": "@muted"}, "axisLabel": {"color": "@muted"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
            "yAxis": {"type": "value", "show": False, "min": -1, "max": 1}, "series": series}


val_main = {
    "id": "valuation_strip", "role": "main", "title": "Where the stock sits among peers", "subtitle": "EV/EBITDA on last full-year EBITDA; peers at their current prices",
    "option": strip_option(29.7, "Welspun at today's price: 29.7x", 16.6),
    "alt": "Dot strip. Peers sit between 11.4x and 26.9x EV/EBITDA with a median of 17.8x. Welspun is at 29.7x at today's price, above every peer; at the report's price of about Rs 129 it was 16.6x.",
    "datatable": table(["Company", "EV/EBITDA x"], [[n, v] for n, v in peers] + [["Welspun, today's price", 29.7], ["Welspun, report price", 16.6]]),
    "variants": [{"key": "today", "label": "At today's price", "option": strip_option(29.7, "Welspun at today's price: 29.7x", 16.6)},
                 {"key": "report", "label": "At report price", "option": strip_option(16.6, "Welspun at the report's price: 16.6x")}],
    "default_variant": "today", "static_note": "At the report's price of about Rs 129, EV/EBITDA was 16.6x.", "height": 230}
pe_band = {
    "id": "pe_history", "role": "small", "title": "P/E against its own history", "subtitle": "Band is the five-year range of 18x to 37x",
    "option": {"grid": {"left": 12, "right": 24, "top": 30, "bottom": 40, "containLabel": True},
               "xAxis": {"type": "value", "min": 0, "max": 130, "name": "P/E, x", "nameLocation": "middle", "nameGap": 26, "nameTextStyle": {"color": "@muted"}, "axisLabel": {"color": "@muted"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
               "yAxis": {"type": "value", "show": False, "min": -1, "max": 1},
               "series": [{"type": "scatter", "symbolSize": 16, "data": [
                   {"value": [116.9, 0.25], "tip": "P/E on FY26 EPS at today's price: 116.9x", "itemStyle": {"color": "@accent"}, "label": {"show": True, "position": "top", "formatter": "FY26 EPS 116.9x"}},
                   {"value": [54.5, -0.25], "tip": "P/E on TTM EPS at today's price: 54.5x", "itemStyle": {"color": "@accent"}, "label": {"show": True, "position": "bottom", "formatter": "TTM EPS 54.5x"}}],
                   "markArea": {"silent": True, "itemStyle": {"color": "@grey5", "opacity": 0.6}, "label": {"show": True, "formatter": "Own range 18x to 37x", "color": "@muted", "position": "insideTop"}, "data": [[{"xAxis": 18}, {"xAxis": 37}]]}}]},
    "alt": "P/E against history. The company's five-year P/E range is 18x to 37x. Today it is 54.5x on trailing 12-month EPS and 116.9x on FY26 EPS.",
    "datatable": table(["Measure", "P/E x"], [["Own range low", 18], ["Own range high", 37], ["TTM EPS", 54.5], ["FY26 EPS", 116.9]]),
    "variants": None, "default_variant": None, "static_note": None, "height": 220}
ruler = {
    "id": "price_ruler", "role": "small", "title": "Fair value per share against Rs 249", "subtitle": "Bands from each forecast case",
    "option": {"grid": {"left": 12, "right": 24, "top": 16, "bottom": 36, "containLabel": True},
               "xAxis": {"type": "value", "min": 100, "max": 320, "name": "Rs per share", "nameLocation": "middle", "nameGap": 24, "nameTextStyle": {"color": "@muted"}, "axisLabel": {"color": "@muted"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
               "yAxis": {"type": "category", "inverse": True, "data": ["Company model", "Peer dataset"], "axisLabel": {"color": "@text"}},
               "series": [{"type": "bar", "stack": "r", "silent": True, "itemStyle": {"color": "transparent"}, "data": [124, 105]},
                          {"type": "bar", "stack": "r", "barWidth": 18, "data": [
                              {"value": 62, "tip": "Company model, 12x to 18x FY27E EBITDA: Rs 124 to 186", "itemStyle": {"color": "@grey3"}, "label": {"show": True, "formatter": "Rs 124 to 186"}},
                              {"value": 50, "tip": "Peer dataset projection, 12x to 18x: Rs 105 to 155", "itemStyle": {"color": "@grey4"}, "label": {"show": True, "formatter": "Rs 105 to 155"}}],
                           "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": "@accent", "width": 2}, "label": {"formatter": "Price Rs 249", "color": "@accent", "position": "end"}, "data": [{"xAxis": 249}]}}]},
    "alt": "Price ruler. Fair value bands are Rs 124 to 186 per share on the company model and Rs 105 to 155 on the peer dataset projection. Today's price of Rs 249 is above both.",
    "datatable": table(["Case", "Low Rs", "High Rs"], [["Company model", 124, 186], ["Peer dataset", 105, 155]]),
    "variants": None, "default_variant": None, "static_note": None, "height": 200}
valuation = {
    "id": "valuation", "n": 5, "nav": "Valuation", "question": "Is the price justified?",
    "headline": "Priced above its history and steadiest peer.",
    "charts": [val_main, pe_band, ruler],
    "chips": [
        chip("pe_vs_history", "P/E against own history", 54.5, "54.5x", "x", "TTM", "concern", "own_history",
             {"min": 18, "max": 37, "peer_median": None, "higher_is_better": False, "value": 54.5},
             "The price asks investors to pay well above what they have paid in the past for each rupee of profit.", calc="Price / TTM EPS at the snapshot price against P/E on year-end figures FY22 to FY26."),
        chip("ev_ebitda_vs_peers", "EV/EBITDA against peers", 29.7, "29.7x", "x", "FY26", "concern", "peers",
             {"min": 11.4, "max": 26.9, "peer_median": 17.8, "higher_is_better": False, "value": 29.7},
             "The market pays more for each rupee of operating profit than for any peer.", calc="(Market cap + net debt) / FY26 EBITDA at today's price; peer median at their current prices."),
        chip("fair_value_vs_price", "Fair value against price", 165, "Rs 165", "", "FY27E", "concern", "own_history",
             {"min": 124, "max": 186, "peer_median": None, "higher_is_better": True, "value": 165},
             "Every forecast case values the shares below today's price, so the price assumes more than the model does.", source="calculated",
             calc="16x model FY27E EBITDA less net debt, per share; range 12x to 18x.")],
    "extra": {"fair_value_cases": [
        {"key": "model", "label": "Company model", "basis": "12x to 18x FY27E EBITDA of Rs 977 cr", "low_price": 124, "high_price": 186, "vs_price_pct_low": -50, "vs_price_pct_high": -25},
        {"key": "peer_dataset", "label": "Peer dataset projection", "basis": "12x to 18x FY27 EBITDA of Rs 780 cr", "low_price": 105, "high_price": 155, "vs_price_pct_low": -58, "vs_price_pct_high": -38}],
        "price": 249.0, "toggle_note": "At the report's price of about Rs 129, EV/EBITDA was 16.6x."},
    "notes": []}

# ----------------------------------------------------------------------------- panel 6
random.seed(7)


def price_series(days):
    px, out = 150.0, []
    for i in range(520):
        px *= 1 + random.gauss(0.0016, 0.017)
        out.append(round(px, 1))
    out = out[-days:]
    k = 249.0 / out[-1]
    return [round(x * k, 1) for x in out]


def sma(a, n):
    return [None if i + 1 < n else round(sum(a[i + 1 - n:i + 1]) / n, 1) for i in range(len(a))]


def price_option(days, step):
    full = price_series(520)
    s50, s200 = sma(full, 50), sma(full, 200)
    sl = slice(len(full) - days, None, step)
    labels = [f"D{len(full) - days + i}" for i in range(0, days, step)]
    close, a50, a200 = full[sl], s50[sl], s200[sl]
    vol = [round(abs(random.gauss(1, 0.4)), 2) for _ in close]
    return {"grid": [{"left": 44, "right": 16, "top": 14, "height": "58%"}, {"left": 44, "right": 16, "top": "74%", "height": "16%"}],
            "tooltip": {"trigger": "axis"},
            "xAxis": [{"type": "category", "data": labels, "gridIndex": 0, "axisLabel": {"show": False}, "axisLine": {"lineStyle": {"color": "@grey4"}}},
                      {"type": "category", "data": labels, "gridIndex": 1, "axisLabel": {"show": False}}],
            "yAxis": [{"type": "value", "scale": True, "gridIndex": 0, "axisLabel": {"color": "@muted"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
                      {"type": "value", "gridIndex": 1, "axisLabel": {"color": "@muted", "formatter": "{value}M"}, "splitLine": {"show": False}}],
            "series": [{"name": "Price", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0, "symbol": "none", "lineStyle": {"color": "@accent", "width": 2},
                        "data": close, "markPoint": {"symbol": "pin", "symbolSize": 46, "itemStyle": {"color": "@accent"}, "label": {"color": "@bg", "formatter": "249", "fontSize": 10},
                                                     "data": [{"coord": [labels[-1], 249.0], "tip": "Latest close Rs 249"}]}},
                       {"name": "50-day average", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0, "symbol": "none", "lineStyle": {"color": "@grey1", "width": 1.5}, "data": a50,
                        "endLabel": {"show": True, "formatter": "50-day", "color": "@grey1"}},
                       {"name": "200-day average", "type": "line", "xAxisIndex": 0, "yAxisIndex": 0, "symbol": "none", "lineStyle": {"color": "@grey3", "width": 1.5, "type": "dashed"}, "data": a200,
                        "endLabel": {"show": True, "formatter": "200-day", "color": "@grey3"}},
                       {"name": "Volume", "type": "bar", "xAxisIndex": 1, "yAxisIndex": 1, "itemStyle": {"color": "@grey4"}, "data": vol}]}, labels, close


o6m, l6m, c6m = price_option(125, 5)
o1y, l1y, c1y = price_option(250, 5)
o2y, l2y, c2y = price_option(500, 5)
month_rows = [[f"Month {i + 1}", c1y[min(i * 4, len(c1y) - 1)]] for i in range(12)]
price_main = {
    "id": "price_trend", "role": "main", "title": "One year of price", "subtitle": "With 50- and 200-day averages and volume",
    "option": o1y,
    "alt": "Line chart of one year of share price with 50- and 200-day averages and volume bars. The price is above both averages and ended at Rs 249 near its 52-week high.",
    "datatable": table(["Period", "Close Rs"], month_rows),
    "variants": [{"key": "6m", "label": "6 months", "option": o6m}, {"key": "1y", "label": "1 year", "option": o1y}, {"key": "2y", "label": "2 years", "option": o2y}],
    "default_variant": "1y", "static_note": "Shown: one year. Six-month and two-year views are in the on-screen report.", "height": 320}
macd_hist = [round(math.sin(i / 3.0) * 4 + (i / 20.0), 2) for i in range(60)]
price_small = {
    "id": "macd", "role": "small", "title": "MACD histogram", "subtitle": "Momentum is fading while price makes new highs",
    "option": {"grid": {"left": 40, "right": 16, "top": 14, "bottom": 22},
               "xAxis": {"type": "category", "data": [str(i) for i in range(60)], "axisLabel": {"show": False}, "axisLine": {"lineStyle": {"color": "@grey4"}}},
               "yAxis": {"type": "value", "axisLabel": {"color": "@muted"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
               "series": [{"type": "bar", "data": [{"value": v, "itemStyle": {"color": "@grey3" if v >= 0 else "@grey4"}} for v in macd_hist],
                           "markLine": {"silent": True, "symbol": "none", "lineStyle": {"color": "@concern", "type": "dashed"}, "label": {"formatter": "Lower high", "color": "@muted"},
                                        "data": [[{"coord": ["24", 6.0]}, {"coord": ["52", 4.6]}]]}}]},
    "alt": "MACD histogram over 60 sessions. The latest peak is lower than the earlier one while price made a higher high, a bearish divergence.",
    "datatable": table(["Window", "Histogram high"], [["Earlier peak", 11.77], ["Latest peak", 9.07]]),
    "variants": None, "default_variant": None, "static_note": None, "height": 200}
price = {
    "id": "price", "n": 6, "nav": "Price", "question": "What is the share price doing?",
    "headline": "Strong trend, losing force.",
    "charts": [price_main, price_small],
    "chips": [chip("stretch", "Stretch above 200-day average", 52.4, "52.4%", "%", "latest", "concern", "general_norm",
                   {"min": 0, "max": 40, "peer_median": None, "higher_is_better": False, "value": 52.4},
                   "The price is far above its long-run average, which has often been followed by a pause.", source="general_norm",
                   calc="Price / 200-day average - 1. General norm: above 30% is stretched.")],
    "extra": {
        "tests": [{"key": "trend", "label": "Trend", "measures": "Price vs 50/200 averages, crossover", "state": "Up", "verdict": "good", "evidence": "Price above both averages; golden cross on 1 June."},
                  {"key": "momentum", "label": "Momentum", "measures": "RSI and MACD combined, counted once", "state": "Up, fading", "verdict": "neutral", "evidence": "RSI 58.9; MACD histogram below its earlier peak."},
                  {"key": "participation", "label": "Participation", "measures": "Volume in this trend vs the prior base", "state": "Weak", "verdict": "concern", "evidence": "Median volume is 0.18x the prior base."},
                  {"key": "stretch", "label": "Stretch", "measures": "% above the 200-day average", "state": "Very high", "verdict": "concern", "evidence": "52.4% above the 200-day average."}],
        "verdict_sentence": "Uptrend intact, but late, stretched and thinly traded.",
        "divergence": {"sentence": "Bearish divergence: price made a higher high while MACD made a lower high (11.77 to 9.07)."},
        "rsi_zone": "RSI 58.9 is in the neutral zone (30 to 60).",
        "readings": [["Price", "Rs 249.00", "close on the snapshot date"], ["52-week range", "Rs 107.40 to Rs 267.40", "88.5% of the range"],
                     ["200-day average", "Rs 163.43", "price +52.4%"], ["RSI (14)", "58.9", "neutral"]]},
    "notes": []}

# ----------------------------------------------------------------------------- panel 7
watch_main = {
    "id": "signposts", "role": "main", "title": "Where each signpost stands", "subtitle": "Current value against the level that would confirm recovery",
    "option": {"grid": {"left": 150, "right": 50, "top": 10, "bottom": 24},
               "xAxis": {"type": "value", "axisLabel": {"color": "@muted"}, "splitLine": {"lineStyle": {"color": "@grid"}}},
               "yAxis": {"type": "category", "inverse": True, "data": ["SG&A, % of revenue (want below 34)", "EBITDA margin, % (want above 12)", "Interest cover, x (want above 4)"],
                         "axisLabel": {"color": "@text", "width": 140, "overflow": "break"}},
               "series": [{"name": "Now", "type": "bar", "barWidth": 14, "data": [
                              {"value": 37.8, "tip": "SG&A 37.8% of revenue now; target below 34%", "itemStyle": {"color": "@accent"}, "label": {"show": True, "position": "right", "formatter": "37.8"}},
                              {"value": 10.3, "tip": "TTM EBITDA margin 10.3%; target above 12%", "itemStyle": {"color": "@accent"}, "label": {"show": True, "position": "right", "formatter": "10.3"}},
                              {"value": 2.8, "tip": "Interest cover 2.8x; target above 4x", "itemStyle": {"color": "@accent"}, "label": {"show": True, "position": "right", "formatter": "2.8"}}]},
                          {"name": "Target", "type": "bar", "barWidth": 14, "data": [
                              {"value": 34, "tip": "Target: below 34%", "itemStyle": {"color": "@grey4"}, "label": {"show": True, "position": "right", "formatter": "34"}},
                              {"value": 12, "tip": "Target: above 12%", "itemStyle": {"color": "@grey4"}, "label": {"show": True, "position": "right", "formatter": "12"}},
                              {"value": 4, "tip": "Target: above 4x", "itemStyle": {"color": "@grey4"}, "label": {"show": True, "position": "right", "formatter": "4"}}]}]},
    "alt": "Bar chart of three signposts against their targets: SG&A 37.8% against a target below 34%, EBITDA margin 10.3% against above 12%, and interest cover 2.8x against above 4x.",
    "datatable": table(["Signpost", "Now", "Target"], [["SG&A, % of revenue", 37.8, "below 34"], ["EBITDA margin, %", 10.3, "above 12"], ["Interest cover, x", 2.8, "above 4"]]),
    "variants": None, "default_variant": None, "static_note": None, "height": 210}
watch = {
    "id": "watch", "n": 7, "nav": "Watch", "question": "What would change the view?",
    "headline": "Three signs the recovery is real.",
    "charts": [watch_main],
    "chips": [chip("ttm_ebitda", "TTM EBITDA against last full year", 1068, "Rs 1,068 cr", "", "TTM", "good", "own_history",
                   {"min": 753, "max": 1369, "peer_median": None, "higher_is_better": True, "value": 1068},
                   "Recent quarters are already earning more than the full year, which is the first sign of recovery.", source="calculated",
                   calc="Sum of the last four reported quarters of EBITDA.")],
    "extra": {"signposts": [
        {"metric": "SG&A as % of revenue", "current": 37.8, "current_text": "37.8%", "threshold_text": "Below 34% of revenue", "status": "concern", "why": "Overheads must fall back for margins to recover."},
        {"metric": "EBITDA margin", "current": 10.3, "current_text": "10.3% (TTM)", "threshold_text": "Above 12% for two to three quarters", "status": "neutral", "why": "A sustained margin above 12% would confirm the forecast path."},
        {"metric": "Interest coverage", "current": 2.8, "current_text": "2.8x", "threshold_text": "Above 4x", "status": "concern", "why": "Thin cover leaves little room for another weak year."},
        {"metric": "Operating cash flow / EBITDA", "current": 100, "current_text": "148% in FY26", "threshold_text": "Near 80% while sales grow", "status": "neutral", "why": "Cash should hold up even when working capital absorbs it."}],
        "portfolio": {"current_weight": None, "target_weight": None, "triggers": None}},
    "notes": []}

model = {
    "schema_version": 1,
    "meta": {"ticker": "WELSPUNLIV", "company_name": "Welspun Living Ltd", "sector": "Home textiles", "kind": "nonfinancial", "report_date": "2026-11-15",
             "generated_at": "2026-11-15T10:27:30", "currency": "INR",
             "snapshot": {"price": 249.0, "price_date": "2026-11-13", "price_time": "15:30 IST", "market_cap_cr": 23524, "week52_low": 107.4, "week52_high": 267.4}},
    "thesis": "A sound product business, hurt by fixed costs, priced for a recovery that has only started.",
    "warnings": [{"level": "warning", "text": "Peer EV/EBITDA is on each company's current price; the history uses fiscal-year average prices."},
                 {"level": "info", "text": "Q2 FY27 is in the TTM figures but not in the FY26 full-year figures."}],
    "panels": [summary, profit, cyc, strength, valuation, price, watch],
    "glossary": [
        {"id": "ev_ebitda", "term": "EV/EBITDA", "definition": "Enterprise value divided by operating profit before interest, tax, depreciation and amortisation. It shows what investors pay for each rupee of operating profit."},
        {"id": "ttm", "term": "TTM", "definition": "Trailing twelve months: the last four reported quarters added together."},
        {"id": "sga", "term": "SG&A", "definition": "Selling, general and administrative costs: overheads that do not change directly with each unit sold."},
        {"id": "eps", "term": "EPS", "definition": "Earnings per share: profit after tax divided by the number of shares."},
        {"id": "rsi", "term": "RSI", "definition": "Relative strength index, a 0 to 100 gauge of recent price momentum. Above 70 is overbought, below 30 oversold."},
        {"id": "macd", "term": "MACD", "definition": "A momentum gauge built from two moving averages of price."},
        {"id": "p_e", "term": "P/E", "definition": "Price divided by earnings per share."}],
    "sources": {"provider": "FinEdge API", "snapshot": "Price as of the 13 Nov 2026 close; market cap from the exchange quote.",
                "definitions": ["Net debt is borrowings less cash and current investments at the latest balance sheet.", "Enterprise value is market cap plus net debt."],
                "validation": {"errors": [], "warnings": ["news_summary: per-share figure Rs 261.95 comes from a news item"], "passed": ["32 reconciliation checks", "No factor-of-10 unit slips"]}},
    "disclaimer": "This report is for information only and is not investment advice. Figures come from company filings and the data provider and may contain errors."}

if __name__ == "__main__":
    path = os.path.join(HERE, "stub.report.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(model, fh, indent=1, ensure_ascii=False)
    print(path, os.path.getsize(path), "bytes")
