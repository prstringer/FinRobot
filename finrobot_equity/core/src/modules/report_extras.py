#!/usr/bin/env python
# coding: utf-8
"""
Rendering of the interpretation layer: summary panel, verdict chips, TTM table, technical analysis, cash flow and
ratio bars. Everything is drawn from the facts saved in ``report_context.json`` (the analysis step computes them once),
so HTML and PDF show the same numbers. Every value printed in a chart also appears in a table or sentence, so the PDF
is readable by text extraction.
"""

from __future__ import annotations

import base64
import os
from typing import Optional

import pandas as pd

from . import cashflow_analysis as ca
from . import insights as ins_mod
from . import ratio_bars as rb
from . import technical_analysis as ta

COLORS = {"good": "#059669", "neutral": "#d97706", "concern": "#dc2626"}
BG = {"good": "#ecfdf5", "neutral": "#fffbeb", "concern": "#fef2f2"}
LIGHT_COLOR = {"green": "#059669", "amber": "#d97706", "red": "#dc2626"}
WORD = {"good": "Good", "neutral": "Neutral", "concern": "Concern"}
YARD = {"own_history": "own history", "peers": "peers", "general_norm": "general norm (not company-specific)"}


def _data_uri(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


# ----------------------------------------------------------------------------------------------
# asset pack
# ----------------------------------------------------------------------------------------------
def build_pack(ctx: Optional[dict], analysis_dir: str, out_dir: str, ticker: str) -> dict:
    """Charts (as data URIs) and table rows for every interpretation section. Missing inputs just omit that section."""
    pack: dict = {}
    if not ctx:
        return pack
    ins = ctx.get("insights") or {}
    ex = ctx.get("extras") or {}
    pack["summary_panel"] = ins.get("summary_panel")
    pack["verdicts"] = list((ins.get("verdicts") or {}).values())
    pack["ttm"] = ins.get("ttm_comparison")
    pack["peer_statements"] = (ins.get("peer_statements") or {}).get("sentences") or []

    # technical
    tf_json = ex.get("technical")
    price_path = os.path.join(analysis_dir, "historical_price_full.csv")
    if tf_json and tf_json.get("available", True) and os.path.exists(price_path):
        try:
            from .technical_charts import make_technical_chart
            price_df = pd.read_csv(price_path)
            eps = (tf_json.get("price_vs_earnings") or {})
            info = {"eps_fy_now": eps.get("eps_fy_now"), "eps_fy_prev": eps.get("eps_fy_prev"), "eps_ttm": eps.get("eps_ttm")}
            tf = ta.compute_technical_facts(price_df, info)
            path = make_technical_chart(price_df, tf, os.path.join(out_dir, f"{ticker}_technical_analysis.png"), ticker)
            pack["technical"] = {"rows": ta.readings_table_rows(tf), "tests": ta.tests_rows(tf), "verdict": tf.get("verdict_sentence"),
                                 "divergence": tf.get("divergence_sentence"), "pve": (tf.get("price_vs_earnings") or {}),
                                 "chart": _data_uri(path)}
        except Exception as e:
            print(f"Warning: technical section skipped: {e}")

    # cash flow
    cf = ex.get("cashflow")
    if cf and cf.get("available", True):
        try:
            from .cashflow_charts import draw_cashflow_charts
            paths = draw_cashflow_charts(cf, out_dir, ticker)
            header, rows = ca.cashflow_table_rows(cf)
            pack["cashflow"] = {"header": header, "rows": rows, "answers": cf.get("answers") or {}, "lines": ca.cashflow_text_lines(cf),
                                "cyclicality": (cf.get("cyclicality") or {}).get("statement") if (cf.get("cyclicality") or {}).get("flag") else None,
                                "capex_basis": cf.get("capex_basis"), "unavailable": cf.get("unavailable") or [],
                                "charts": {k: _data_uri(v) for k, v in (paths or {}).items()}}
        except Exception as e:
            print(f"Warning: cash-flow section skipped: {e}")

    # ratio bars (replace the radar chart)
    rbf = ex.get("ratio_bars")
    if rbf:
        try:
            path = rb.draw_ratio_bars(rbf, os.path.join(out_dir, f"{ticker}_ratio_bars.png"))
            pack["ratio_bars"] = {"rows": rb.ratio_bar_rows(rbf), "chart": _data_uri(path)}
        except Exception as e:
            print(f"Warning: ratio bars skipped: {e}")
    return pack


# ----------------------------------------------------------------------------------------------
# HTML
# ----------------------------------------------------------------------------------------------
def _pos(scale: dict, v: Optional[float]) -> Optional[float]:
    lo, hi = scale.get("range_min"), scale.get("range_max")
    if v is None or lo is None or hi is None or hi == lo:
        return None
    return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))


def scale_bar_html(scale: dict, verdict: str) -> str:
    p = _pos(scale, scale.get("value"))
    pm = _pos(scale, scale.get("peer_median")) if scale.get("peer_median") is not None else None
    if p is None:
        return ""
    tick = f'<div title="peer median" style="position:absolute;left:{pm:.0f}%;top:-3px;width:2px;height:12px;background:#334155;"></div>' if pm is not None else ""
    return ('<div style="position:relative;height:6px;background:#e2e8f0;border-radius:3px;margin:8px 4px 2px 4px;min-width:120px;">'
            f'{tick}<div style="position:absolute;left:{p:.0f}%;top:-3px;width:12px;height:12px;margin-left:-6px;border-radius:50%;background:{COLORS[verdict]};border:2px solid #fff;box-shadow:0 0 0 1px {COLORS[verdict]};"></div></div>')


def chip_html(v: dict) -> str:
    c = COLORS[v["verdict"]]
    return f'<span style="display:inline-block;background:{c};color:#fff;font-weight:700;font-size:0.7rem;padding:1px 8px;border-radius:9999px;">{WORD[v["verdict"]]}</span>'


def summary_panel_html(pack: dict) -> str:
    panel = pack.get("summary_panel")
    if not panel:
        return ""
    rows = ""
    for q in panel:
        c = LIGHT_COLOR.get(q["light"], "#64748b")
        rows += (f'<tr><td style="width:34px;text-align:center;vertical-align:top;"><span style="display:inline-block;width:16px;height:16px;border-radius:50%;background:{c};"></span></td>'
                 f'<td style="vertical-align:top;font-weight:600;width:28%;">{q["q"]}</td><td style="vertical-align:top;">{q["answer"]}</td></tr>')
    return ('<section class="mb-10" id="summary"><h2 class="section-title">At a Glance</h2>'
            '<div style="overflow-x:auto;"><table class="data-table"><tbody>' + rows + '</tbody></table></div>'
            '<p class="caption" style="font-size:0.7rem;color:#64748b;">Green: healthy. Amber: mixed or watch. Red: concern. Judgements compare each figure with the company\'s own history, its peers, '
            'or a labelled general norm.</p></section>')


def verdict_table_html(pack: dict) -> str:
    vs = pack.get("verdicts") or []
    if not vs:
        return ""
    body = ""
    for v in vs:
        s = v["scale"]
        rng = (f'{s["range_min"]:.1f} to {s["range_max"]:.1f}' if s.get("range_min") is not None else "n/a")
        peer = f', peer median {s["peer_median"]:.1f}' if s.get("peer_median") is not None else ""
        body += (f'<tr><td style="vertical-align:top;"><b>{v["label"]}</b><br><span style="font-size:0.7rem;color:#64748b;">{s["value"]:.1f}{v.get("unit","")} '
                 f'(range {rng}{peer}); {v["direction"].replace("_", " ")}</span></td>'
                 f'<td style="vertical-align:top;">{chip_html(v)}<br><span style="font-size:0.65rem;color:#64748b;">vs {YARD.get(v["yardstick"], v["yardstick"])}</span></td>'
                 f'<td style="vertical-align:top;min-width:140px;">{scale_bar_html(s, v["verdict"])}</td>'
                 f'<td style="vertical-align:top;">{v["so_what"]}<br><span style="font-size:0.65rem;color:#94a3b8;">{v["source_note"]}</span></td></tr>')
    return ('<h3 class="heading-2">Key Metrics: Verdicts</h3><div style="overflow-x:auto;margin-bottom:1rem;"><table class="data-table"><thead><tr>'
            '<th>Metric</th><th>Verdict</th><th>Where it sits</th><th>So what</th></tr></thead><tbody>' + body + '</tbody></table></div>'
            '<p class="caption" style="font-size:0.7rem;color:#64748b;">The dot shows the current value inside its own five-year range; the dark tick marks the peer median where available.</p>')


def ttm_table_html(pack: dict) -> str:
    t = pack.get("ttm")
    if not t:
        return ""
    body = ""
    for r in t["rows"]:
        chg = "" if r["change_pct"] is None else (f'{r["change_pct"]:+.1f} pts' if r["unit"] == "pts" else f'{r["change_pct"]:+.0f}%')
        body += f'<tr><td>{r["label"]}</td><td style="text-align:right;">{r["fy"]:,.{2 if r["unit"] == "Rs" else 1}f}</td><td style="text-align:right;">{r["ttm"]:,.{2 if r["unit"] == "Rs" else 1}f}</td><td style="text-align:right;">{chg}</td></tr>'
    note = f'<p class="body-text">{t["text"]}</p>' if t.get("text") else ""
    return (f'<h3 class="heading-2">Last Full Year versus Trailing Four Quarters</h3><div style="overflow-x:auto;"><table class="data-table"><thead><tr><th>Rs crore unless stated</th>'
            f'<th>{t["fy_label"]}</th><th>TTM to {t["ttm_period_end"]}</th><th>Change</th></tr></thead><tbody>{body}</tbody></table></div>{note}')


def _table(header, rows, right_from=1) -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    tr = "".join("<tr>" + "".join(f'<td{" style=text-align:right;" if i >= right_from else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div style="overflow-x:auto;margin-bottom:1rem;"><table class="data-table"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def _img(src: str, alt: str) -> str:
    return f'<img src="{src}" alt="{alt}" style="max-width:100%;border-radius:0.5rem;margin-bottom:0.75rem;">' if src else ""


def technical_html(pack: dict) -> str:
    t = pack.get("technical")
    if not t:
        return ""
    pve = t["pve"]
    strip = ""
    if pve.get("sentence"):
        strip = ('<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:0.5rem;padding:0.6rem 0.9rem;margin:0.6rem 0;">'
                 f'<b>Price says / earnings say:</b> 1-year price change {pve["price_change_1y_pct"]:+.0f}%; EPS {pve["eps_fy_now"]} against {pve["eps_fy_prev"]} the year before '
                 f'({pve["eps_fy_change_pct"]:+.0f}%); TTM EPS {pve["eps_ttm"]} ({pve["eps_ttm_change_pct"]:+.0f}% against {pve["eps_ttm_base_label"]}). {pve["sentence"]}</div>')
    tests = "".join(f"<tr><td><b>{a}</b></td><td>{b}</td><td><b>{c}</b></td><td>{d}</td></tr>" for a, b, c, d in t["tests"])
    html = '<h3 class="heading-2">Technical Analysis</h3>'
    html += f'<p class="body-text"><b>{t["verdict"]}</b></p>' if t.get("verdict") else ""
    html += strip + _img(t.get("chart", ""), "Price, RSI and MACD with divergence marked")
    html += _table(["Reading", "Value", "Note"], t["rows"], right_from=9)
    html += ('<h4 style="font-weight:600;margin:0.5rem 0;">Four independent tests</h4><div style="overflow-x:auto;margin-bottom:1rem;"><table class="data-table"><thead><tr>'
             '<th>Test</th><th>Measures</th><th>State</th><th>Evidence</th></tr></thead><tbody>' + tests + '</tbody></table></div>')
    html += f'<p class="body-text">{t["divergence"]}</p>' if t.get("divergence") else ""
    return html


def cashflow_html(pack: dict) -> str:
    c = pack.get("cashflow")
    if not c:
        return ""
    html = '<h3 class="heading-2">Cash Flow Analysis</h3>' + _table(c["header"], c["rows"])
    html += f'<p class="caption" style="font-size:0.7rem;color:#64748b;">{c["capex_basis"]}. Money out is in brackets.</p>' if c.get("capex_basis") else ""
    if c.get("unavailable"):
        html += f'<p class="caption" style="font-size:0.7rem;color:#64748b;">Not available from the data source: {"; ".join(c["unavailable"])}.</p>'
    answers = c.get("answers") or {}
    for key, q in (("profit_to_cash", "Does profit turn into cash?"), ("funds_itself", "Does the business fund itself?"),
                   ("flattered_by_one_offs", "Is cash flattered by one-offs?")):
        a = answers.get(key)
        if a:
            col = COLORS[a["verdict"]]
            html += (f'<div style="border-left:4px solid {col};background:{BG[a["verdict"]]};padding:6px 10px;margin:6px 0;"><b>{q}</b> {chip_html(a)} '
                     f'<span style="font-size:0.7rem;color:#64748b;">vs {YARD.get(a["yardstick"], a["yardstick"])}</span><br>{a["so_what"]}'
                     f'<br><span style="font-size:0.65rem;color:#94a3b8;">{a["source_note"]}</span></div>')
    if c.get("cyclicality"):
        html += f'<p class="body-text"><b>Cyclicality:</b> {c["cyclicality"]}</p>'
    diag = [l.strip() for l in c["lines"] if l.strip().startswith(("Five-year", "Working-capital", "Cyclicality"))]
    html += "".join(f'<p class="body-text" style="margin:0.2rem 0;">{l}</p>' for l in diag if not l.startswith("Cyclicality"))
    ch = c.get("charts") or {}
    imgs = "".join(_img(ch.get(k, ""), k) for k in ("ebitda_vs_cfo", "cash_waterfall", "working_capital_days"))
    html += f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">{imgs}</div>' if imgs else ""
    return html


def ratio_bars_html(pack: dict) -> str:
    r = pack.get("ratio_bars")
    if not r:
        return ""
    rows = [[x["label"], x["value"], x["own_range"], x.get("peer_median") or "n/a", f'{x["verdict"].capitalize()} vs {YARD.get(x["yardstick"], x["yardstick"])}', x["so_what"]] for x in r["rows"]]
    return ('<h3 class="heading-2">Financial Ratios against Benchmarks</h3>' + _img(r.get("chart", ""), "Ratio bars")
            + _table(["Ratio", "Now", "Own 5-year range", "Peer median", "Verdict", "So what"], rows, right_from=9))


def analysis_sections_html(pack: dict) -> str:
    """Technical, cash flow and ratio sections (replace the old chart grid)."""
    return technical_html(pack) + cashflow_html(pack) + ratio_bars_html(pack)


# ----------------------------------------------------------------------------------------------
# PDF (ReportLab)
# ----------------------------------------------------------------------------------------------
def pdf_flowables(pack: dict, kind: str, styles, content_width, Paragraph, mm, load_image):
    """Flowables for one section: 'summary', 'verdicts', 'ttm', 'technical', 'cashflow', 'ratios'."""
    from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
    from reportlab.lib import colors
    from reportlab.platypus import KeepTogether, Spacer, Table, TableStyle

    cap, body, h2 = styles["Caption"], styles["Body"], styles["Heading2"]

    def chip(v: str, w=20 * mm, h=5 * mm):
        d = Drawing(w, h)
        d.add(Rect(0, 0, w, h, rx=2.5 * mm, ry=2.5 * mm, fillColor=colors.HexColor(COLORS[v]), strokeColor=None))
        d.add(String(w / 2, 1.4 * mm, WORD[v], textAnchor="middle", fontName="Helvetica-Bold", fontSize=7, fillColor=colors.white))
        return d

    def bar(scale: dict, v: str, w=36 * mm, h=5 * mm):
        d = Drawing(w, h)
        d.add(Rect(0, h / 2 - 0.6 * mm, w, 1.2 * mm, fillColor=colors.HexColor("#e2e8f0"), strokeColor=None))
        pm = _pos(scale, scale.get("peer_median")) if scale.get("peer_median") is not None else None
        if pm is not None:
            d.add(Line(w * pm / 100, 0.3 * mm, w * pm / 100, h - 0.3 * mm, strokeColor=colors.HexColor("#334155"), strokeWidth=1))
        p = _pos(scale, scale.get("value"))
        if p is not None:
            d.add(Circle(w * p / 100, h / 2, 1.6 * mm, fillColor=colors.HexColor(COLORS[v]), strokeColor=colors.white, strokeWidth=0.6))
        return d

    def light(color: str):
        d = Drawing(5 * mm, 5 * mm)
        d.add(Circle(2.5 * mm, 2.5 * mm, 2.2 * mm, fillColor=colors.HexColor(LIGHT_COLOR.get(color, "#64748b")), strokeColor=None))
        return d

    def tbl(data, widths, header=True, zebra=True):
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
              ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
        if header:
            st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0"))]
        if zebra:
            st += [("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")])]
        t.setStyle(TableStyle(st))
        return t

    P = lambda s, st=cap: Paragraph(str(s), st)
    out = []

    if kind == "summary" and pack.get("summary_panel"):
        data = [[light(q["light"]), P(f"<b>{q['q']}</b>"), P(q["answer"], body)] for q in pack["summary_panel"]]
        out += [Paragraph("At a Glance", h2), Spacer(1, 2 * mm), tbl(data, [8 * mm, 42 * mm, content_width - 50 * mm], header=False),
                P("Green: healthy. Amber: mixed or watch. Red: concern. Each judgement compares a figure with the company's own history, its peers, or a labelled general norm."),
                Spacer(1, 4 * mm)]

    if kind == "verdicts" and pack.get("verdicts"):
        data = [[P("<b>Metric</b>"), P("<b>Verdict</b>"), P("<b>Where it sits</b>"), P("<b>So what</b>")]]
        for v in pack["verdicts"]:
            s = v["scale"]
            rng = f'{s["range_min"]:.1f} to {s["range_max"]:.1f}' if s.get("range_min") is not None else "n/a"
            peer = f', peer median {s["peer_median"]:.1f}' if s.get("peer_median") is not None else ""
            data.append([P(f"<b>{v['label']}</b><br/>{s['value']:.1f}{v.get('unit','')} (range {rng}{peer}); {v['direction'].replace('_', ' ')}"),
                         [chip(v["verdict"]), P(f"vs {YARD.get(v['yardstick'], v['yardstick'])}")], bar(s, v["verdict"]),
                         P(f"{v['so_what']}<br/><font size=6 color='#94a3b8'>{v['source_note']}</font>")])
        out += [Paragraph("Key Metrics: Verdicts", h2), Spacer(1, 2 * mm),
                tbl(data, [46 * mm, 26 * mm, 38 * mm, content_width - 110 * mm]),
                P("The dot shows the current value inside its own five-year range; the dark tick marks the peer median where available."), Spacer(1, 4 * mm)]

    if kind == "ttm" and pack.get("ttm"):
        t = pack["ttm"]
        data = [[P("<b>Rs crore unless stated</b>"), P(f"<b>{t['fy_label']}</b>"), P(f"<b>TTM to {t['ttm_period_end']}</b>"), P("<b>Change</b>")]]
        for r in t["rows"]:
            chg = "" if r["change_pct"] is None else (f"{r['change_pct']:+.1f} pts" if r["unit"] == "pts" else f"{r['change_pct']:+.0f}%")
            dec = 2 if r["unit"] == "Rs" else 1
            data.append([P(r["label"]), P(f"{r['fy']:,.{dec}f}"), P(f"{r['ttm']:,.{dec}f}"), P(chg)])
        out += [Paragraph("Last Full Year versus Trailing Four Quarters", h2), Spacer(1, 2 * mm), tbl(data, [60 * mm, 35 * mm, 40 * mm, content_width - 135 * mm])]
        if t.get("text"):
            out += [Spacer(1, 2 * mm), Paragraph(t["text"], body)]
        out.append(Spacer(1, 4 * mm))

    if kind == "technical" and pack.get("technical"):
        t = pack["technical"]
        out += [Paragraph("Technical Analysis", h2), Spacer(1, 2 * mm)]
        if t.get("verdict"):
            out.append(Paragraph(f"<b>{t['verdict']}</b>", body))
        pve = t["pve"]
        if pve.get("sentence"):
            out += [Spacer(1, 2 * mm), Paragraph(
                f"<b>Price says / earnings say:</b> 1-year price change {pve['price_change_1y_pct']:+.0f}%; EPS {pve['eps_fy_now']} against {pve['eps_fy_prev']} the year before "
                f"({pve['eps_fy_change_pct']:+.0f}%); TTM EPS {pve['eps_ttm']} ({pve['eps_ttm_change_pct']:+.0f}% against {pve['eps_ttm_base_label']}). {pve['sentence']}", body)]
        img = load_image(t.get("chart", ""), content_width, content_width * 0.62) if t.get("chart") else None
        if img:
            out += [Spacer(1, 2 * mm), img]
        rows = [[P("<b>Reading</b>"), P("<b>Value</b>"), P("<b>Note</b>")]] + [[P(a), P(f"<b>{b}</b>"), P(c)] for a, b, c in t["rows"]]
        out += [Spacer(1, 2 * mm), tbl(rows, [48 * mm, 32 * mm, content_width - 80 * mm]), Spacer(1, 3 * mm)]
        tests = [[P("<b>Test</b>"), P("<b>Measures</b>"), P("<b>State</b>"), P("<b>Evidence</b>")]] + [[P(f"<b>{a}</b>"), P(b), P(f"<b>{c}</b>"), P(d)] for a, b, c, d in t["tests"]]
        out += [Paragraph("Four independent tests", styles["BodyBold"]), Spacer(1, 1 * mm), tbl(tests, [22 * mm, 40 * mm, 34 * mm, content_width - 96 * mm])]
        if t.get("divergence"):
            out += [Spacer(1, 2 * mm), Paragraph(t["divergence"], body)]
        out.append(Spacer(1, 4 * mm))

    if kind == "cashflow" and pack.get("cashflow"):
        c = pack["cashflow"]
        n = len(c["header"])
        first = 52 * mm
        data = [[P(f"<b>{h}</b>") for h in c["header"]]] + [[P(r[0])] + [P(x) for x in r[1:]] for r in c["rows"]]
        out += [Paragraph("Cash Flow Analysis", h2), Spacer(1, 2 * mm), tbl(data, [first] + [(content_width - first) / (n - 1)] * (n - 1))]
        notes = []
        if c.get("capex_basis"):
            notes.append(c["capex_basis"] + ". Money out is in brackets.")
        if c.get("unavailable"):
            notes.append("Not available from the data source: " + "; ".join(c["unavailable"]) + ".")
        for line in notes:
            out.append(P(line))
        out.append(Spacer(1, 2 * mm))
        answers = c.get("answers") or {}
        qrows = []
        for key, q in (("profit_to_cash", "Does profit turn into cash?"), ("funds_itself", "Does the business fund itself?"),
                       ("flattered_by_one_offs", "Is cash flattered by one-offs?")):
            a = answers.get(key)
            if a:
                qrows.append([P(f"<b>{q}</b>"), [chip(a["verdict"]), P(f"vs {YARD.get(a['yardstick'], a['yardstick'])}")],
                              P(f"{a['so_what']}<br/><font size=6 color='#94a3b8'>{a['source_note']}</font>")])
        if qrows:
            out.append(tbl(qrows, [48 * mm, 32 * mm, content_width - 80 * mm], header=False))
        if c.get("cyclicality"):
            out += [Spacer(1, 2 * mm), Paragraph(f"<b>Cyclicality:</b> {c['cyclicality']}", body)]
        for l in [l.strip() for l in c["lines"] if l.strip().startswith(("Five-year", "Working-capital"))]:
            out.append(Paragraph(l, body))
        ch = c.get("charts") or {}
        w2 = (content_width - 4 * mm) / 2
        imgs = [load_image(ch[k], w2, w2 * 0.62) for k in ("ebitda_vs_cfo", "cash_waterfall") if ch.get(k)]
        imgs = [i for i in imgs if i]
        if imgs:
            out += [Spacer(1, 2 * mm), Table([imgs], colWidths=[w2 + 2 * mm] * len(imgs))]
        wc = load_image(ch["working_capital_days"], w2, w2 * 0.62) if ch.get("working_capital_days") else None
        if wc:
            out += [Spacer(1, 2 * mm), Table([[wc]], colWidths=[w2 + 2 * mm])]
        out.append(Spacer(1, 4 * mm))

    if kind == "ratios" and pack.get("ratio_bars"):
        r = pack["ratio_bars"]
        out += [Paragraph("Financial Ratios against Benchmarks", h2), Spacer(1, 2 * mm)]
        img = load_image(r.get("chart", ""), content_width, content_width * 0.6) if r.get("chart") else None
        if img:
            out += [img, Spacer(1, 2 * mm)]
        data = [[P("<b>Ratio</b>"), P("<b>Now</b>"), P("<b>Own 5-year range</b>"), P("<b>Peer median</b>"), P("<b>Verdict</b>"), P("<b>So what</b>")]]
        for x in r["rows"]:
            data.append([P(x["label"]), P(f"<b>{x['value']}</b>"), P(x["own_range"]), P(x.get("peer_median") or "n/a"),
                         P(f"{x['verdict'].capitalize()} vs {YARD.get(x['yardstick'], x['yardstick'])}"), P(x["so_what"])])
        out += [tbl(data, [34 * mm, 15 * mm, 26 * mm, 17 * mm, 28 * mm, content_width - 120 * mm]), Spacer(1, 4 * mm)]
    return out

def drop_superseded_charts(report_data: dict, pack: dict, output_dir: str, ticker: str) -> None:
    """Remove the legacy radar, cash-flow and technical charts that the interpretation sections replace."""
    superseded = []
    if pack.get("ratio_bars"):
        superseded += [("financial_radar_path", "financial_radar", f"{ticker}_financial_radar.png")]
    if pack.get("cashflow"):
        superseded += [("cash_flow_chart_path", "cash_flow", f"{ticker}_cash_flow_chart.png")]
    if pack.get("technical"):
        superseded += [("technical_indicators_path", "technical_indicators", f"{ticker}_technical_indicators.png"),
                       ("stock_price_chart_path", "stock_price", f"{ticker}_stock_price_chart.png")]
    for key, enh, fname in superseded:
        report_data[key] = ""
        (report_data.get("enhanced_charts") or {}).pop(enh, None)
        try:
            os.remove(os.path.join(output_dir, fname))
        except OSError:
            pass
