#!/usr/bin/env python
# coding: utf-8
"""
Explanatory facts and verdicts, computed in code (never by the language model).

* ``margin_bridge`` / ``operating_leverage`` / ``volatility`` / ``range_position``: why profit moved.
* ``make_verdict``: a verdict object per key metric. Every verdict has a yardstick (own history, peers, or a labelled
  general norm), a direction (higher or lower is better) and a one-sentence ``so_what`` that does not restate the number.
* ``peer_statements``: peer rank, median and above/below sentences generated from the peer table.
* ``ttm_comparison``: last full year against trailing four quarters.
* ``summary_panel``: four questions, each with a traffic light.
* ``fact_cards`` / ``section_guidance``: who owns which fact, so each key fact is stated in full once.

All money is INR crore, margins are percent numbers (8.4 means 8.4%).
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np
import pandas as pd

GENERAL_NORM_LABEL = "general norm, not company-specific"

# direction: +1 higher is better, -1 lower is better
METRIC_DEFS = {
    "contribution_margin": ("Contribution margin", "%", +1),
    "ebitda_margin": ("EBITDA margin", "%", +1),
    "sga_pct": ("SG&A as % of revenue", "%", -1),
    "net_margin": ("Net margin", "%", +1),
    "roe": ("Return on equity", "%", +1),
    "roa": ("Return on assets", "%", +1),
    "interest_coverage": ("Interest coverage (EBIT / interest)", "x", +1),
    "debt_equity": ("Debt to equity", "x", -1),
    "current_ratio": ("Current ratio", "x", +1),
    "revenue_growth": ("Revenue growth", "%", +1),
    "ev_ebitda": ("EV/EBITDA", "x", -1),
    "pe": ("P/E", "x", -1),
}

# general norms used only when neither own history nor peers can be computed: (good if beyond, concern if beyond)
GENERAL_NORMS = {
    "interest_coverage": (4.0, 2.0),
    "debt_equity": (0.5, 1.0),
    "current_ratio": (1.5, 1.0),
    "roe": (15.0, 8.0),
}

SO_WHAT = {
    "contribution_margin": {
        "good": "Pricing and product mix are holding up, so any profit squeeze comes from costs further down.",
        "neutral": "Pricing power is roughly intact, but there is little cushion if input costs rise.",
        "concern": "The business is earning less on each sale before overheads, which points to price or mix pressure.",
    },
    "ebitda_margin": {
        "good": "Operating profit per rupee of sales is strong against its own record, so the business has room to absorb shocks.",
        "neutral": "Operating profitability is average for this company, so earnings depend on sales growth.",
        "concern": "Operating profit per rupee of sales is near its worst, so each fall in sales hurts earnings sharply.",
    },
    "sga_pct": {
        "good": "Overheads are low against the company's own record, which leaves more of each sale as profit.",
        "neutral": "Overheads are in their usual range, so profit mostly follows sales.",
        "concern": "Overheads are taking more of each sale than usual, which is what squeezes profit when sales fall.",
    },
    "net_margin": {
        "good": "A healthy share of sales reaches shareholders after all costs, interest and tax.",
        "neutral": "Shareholders keep an average share of each sale compared with the company's past.",
        "concern": "Little of each sale reaches shareholders, so small cost increases can erase profit.",
    },
    "roe": {
        "good": "Equity is being put to work well, which supports value creation for shareholders.",
        "neutral": "Returns on shareholder money are middling for this company.",
        "concern": "Shareholder money earns a weak return, so the stock needs a recovery to justify its price.",
    },
    "roa": {
        "good": "The asset base is generating strong profit.",
        "neutral": "Asset productivity is typical for this company.",
        "concern": "The asset base earns little, which is a drag on returns.",
    },
    "interest_coverage": {
        "good": "Profit covers interest comfortably, so debt is not a near-term worry.",
        "neutral": "Interest is covered, but a profit dip would quickly tighten the margin of safety.",
        "concern": "Profit barely covers interest, so a further earnings dip would put debt service under strain.",
    },
    "debt_equity": {
        "good": "Borrowing is modest against equity, which gives financial flexibility.",
        "neutral": "Borrowing is moderate, so leverage is manageable but worth monitoring.",
        "concern": "Borrowing is heavy against equity, which raises risk if earnings fall.",
    },
    "current_ratio": {
        "good": "Short-term assets comfortably cover short-term bills.",
        "neutral": "Short-term liquidity is adequate but not generous.",
        "concern": "Short-term assets barely cover short-term bills, so liquidity is tight.",
    },
    "revenue_growth": {
        "good": "Sales are growing faster than the company's usual pace.",
        "neutral": "Sales growth is within its normal range.",
        "concern": "Sales are shrinking or growing well below the company's usual pace, which exposes fixed costs.",
    },
    "ev_ebitda": {
        "good": "The market pays less for each rupee of operating profit than it does for peers, so expectations are modest.",
        "neutral": "The market pays about the same for operating profit as it does for peers.",
        "concern": "The market pays a premium for each rupee of operating profit, so the price already assumes a recovery.",
    },
    "pe": {
        "good": "Earnings are cheap against the company's own history.",
        "neutral": "The price is in line with where the company has usually traded on earnings.",
        "concern": "The price is high against earnings, so it relies on profits recovering fast.",
    },
}


def _f(x: Any) -> Optional[float]:
    try:
        v = float(str(x).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# ----------------------------------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------------------------------
def make_verdict(key: str, value: Optional[float], history: Optional[list], peer_median: Optional[float] = None,
                 label: Optional[str] = None, unit: Optional[str] = None, direction: Optional[int] = None,
                 peer_basis: str = "peers") -> Optional[dict]:
    """Verdict object for one metric, or None when no yardstick exists (no verdict without a yardstick).

    ``history`` is the metric over its own five-year window (current value included). Peers are used when
    ``peer_median`` is given. A general norm is used only when both are unavailable, and is labelled as such.
    """
    name, u, d = METRIC_DEFS.get(key, (label or key, unit or "", direction or +1))
    label, unit, direction = label or name, unit or u, direction or d
    if value is None:
        return None
    hist = [h for h in (history or []) if h is not None]
    scale = {"value": value, "range_min": min(hist) if hist else None, "range_max": max(hist) if hist else None,
             "peer_median": peer_median}
    verdict = yardstick = source = None
    if peer_median is not None and key in ("ev_ebitda",):
        ratio = value / peer_median if peer_median else None
        if ratio is not None:
            better = ratio <= 0.9 if direction < 0 else ratio >= 1.1
            worse = ratio >= 1.2 if direction < 0 else ratio <= 0.9
            verdict = "good" if better else ("concern" if worse else "neutral")
            yardstick, source = "peers", f"report data ({peer_basis})"
    if verdict is None and len(hist) >= 3 and max(hist) > min(hist):
        pos = (value - min(hist)) / (max(hist) - min(hist))
        pos = pos if direction > 0 else 1 - pos
        verdict = "good" if pos >= 0.67 else ("concern" if pos <= 0.33 else "neutral")
        yardstick, source = "own_history", "report data (own five-year range)"
    if verdict is None and len(hist) >= 3 and max(hist) == min(hist):
        verdict, yardstick, source = "neutral", "own_history", "report data (own five-year range)"
    if verdict is None and key in GENERAL_NORMS:
        good_at, bad_at = GENERAL_NORMS[key]
        if direction > 0:
            verdict = "good" if value >= good_at else ("concern" if value < bad_at else "neutral")
        else:
            verdict = "good" if value <= good_at else ("concern" if value > bad_at else "neutral")
        yardstick, source = "general_norm", GENERAL_NORM_LABEL
    if verdict is None:
        return None
    text = SO_WHAT.get(key, {}).get(verdict) or {
        "good": "This is a strength against its yardstick.", "neutral": "This is in line with its yardstick.",
        "concern": "This is a weakness against its yardstick."}[verdict]
    return {"key": key, "label": label, "unit": unit, "direction": "higher_is_better" if direction > 0 else "lower_is_better",
            "verdict": verdict, "yardstick": yardstick, "scale": scale, "so_what": text, "source_note": source}


# ----------------------------------------------------------------------------------------------
# series helpers
# ----------------------------------------------------------------------------------------------
def _row_series(final_df: pd.DataFrame, metric: str, suffix: str = "A") -> dict[str, float]:
    m = final_df["metrics"].astype(str)
    sel = final_df[(m == metric) | m.str.startswith(metric + " (")]
    if sel.empty:
        return {}
    out = {}
    for c in final_df.columns:
        if isinstance(c, str) and re.fullmatch(rf"\d{{4}}{suffix}", c):
            v = _f(sel[c].iloc[0])
            if v is not None:
                out[c] = v
    return dict(sorted(out.items()))


def _ratio_pct(num: dict, den: dict) -> dict[str, float]:
    return {y: num[y] / den[y] * 100 for y in num if y in den and den[y]}


def margin_series(final_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Contribution margin, SG&A % and EBITDA margin from the absolute rows (more precise than the rounded % rows)."""
    rev, cp, sga, ebitda = (_row_series(final_df, k) for k in ("Revenue", "Contribution Profit", "SG&A", "EBITDA"))
    return {"contribution_margin": _ratio_pct(cp, rev), "sga_pct": _ratio_pct(sga, rev), "ebitda_margin": _ratio_pct(ebitda, rev),
            "revenue": rev, "ebitda": ebitda}


def margin_bridge(final_df: pd.DataFrame, y0: Optional[str] = None, y1: Optional[str] = None) -> Optional[dict]:
    """Change in EBITDA margin split into the change in contribution margin and the change in SG&A %."""
    ms = margin_series(final_df)
    years = sorted(set(ms["contribution_margin"]) & set(ms["sga_pct"]) & set(ms["ebitda_margin"]))
    if len(years) < 2:
        return None
    y0, y1 = y0 or years[0], y1 or years[-1]
    d_cm = ms["contribution_margin"][y1] - ms["contribution_margin"][y0]
    d_sga = ms["sga_pct"][y1] - ms["sga_pct"][y0]
    d_em = ms["ebitda_margin"][y1] - ms["ebitda_margin"][y0]
    return {"from": y0, "to": y1, "ebitda_margin_from": ms["ebitda_margin"][y0], "ebitda_margin_to": ms["ebitda_margin"][y1],
            "d_ebitda_margin_pts": d_em, "d_contribution_margin_pts": d_cm, "d_sga_pct_pts": d_sga,
            "residual_pts": d_em - (d_cm - d_sga),
            "text": (f"EBITDA margin moved {d_em:+.1f} points from {y0} to {y1} ({ms['ebitda_margin'][y0]:.1f}% to {ms['ebitda_margin'][y1]:.1f}%): "
                     f"contribution margin {d_cm:+.1f} points, SG&A as % of revenue {d_sga:+.1f} points.")}


def operating_leverage(final_df: pd.DataFrame) -> Optional[dict]:
    """EBITDA change per 100 of revenue change between the last two actual years."""
    ms = margin_series(final_df)
    years = sorted(set(ms["revenue"]) & set(ms["ebitda"]))
    if len(years) < 2:
        return None
    a, b = years[-2], years[-1]
    d_rev, d_e = ms["revenue"][b] - ms["revenue"][a], ms["ebitda"][b] - ms["ebitda"][a]
    if not d_rev:
        return None
    per100 = d_e / d_rev * 100
    rv = "fell" if d_rev < 0 else "rose"
    ev = "fell" if d_e < 0 else "rose"
    word = "lost" if d_rev < 0 and d_e < 0 else ("gained" if d_rev > 0 and d_e > 0 else "moved")
    return {"from": a, "to": b, "d_revenue_cr": d_rev / 1e7, "d_ebitda_cr": d_e / 1e7, "ebitda_per_100_revenue": per100,
            "text": (f"From {a} to {b}, revenue {rv} by Rs {abs(d_rev)/1e7:,.0f} Cr and EBITDA {ev} by Rs {abs(d_e)/1e7:,.0f} Cr, "
                     f"so about Rs {abs(per100):.0f} of EBITDA was {word} for every Rs 100 of revenue {'lost' if d_rev < 0 else 'gained'}.")}


def volatility(final_df: pd.DataFrame) -> Optional[dict]:
    ms = margin_series(final_df)
    growth = _row_series(final_df, "Revenue Growth")
    out = {}
    for k, s in (("revenue_growth", growth), ("contribution_margin", ms["contribution_margin"]),
                 ("sga_pct", ms["sga_pct"]), ("ebitda_margin", ms["ebitda_margin"])):
        if len(s) >= 3:
            out[k] = {"min": min(s.values()), "max": max(s.values()), "range_pts": max(s.values()) - min(s.values())}
    if "revenue_growth" not in out or len(out) < 2:
        return None
    rg = out["revenue_growth"]["range_pts"]
    out["text"] = (f"Revenue growth swung {rg:.0f} points over the window; " + "; ".join(
        f"{METRIC_DEFS[k][0]} moved {v['range_pts']:.1f}" for k, v in out.items() if k in ("contribution_margin", "sga_pct", "ebitda_margin")) + " points.")
    return out


def range_position(value: Optional[float], series: list[float]) -> Optional[dict]:
    s = [x for x in series if x is not None]
    if value is None or len(s) < 2 or max(s) == min(s):
        return None
    pct = (value - min(s)) / (max(s) - min(s)) * 100
    return {"value": value, "min": min(s), "max": max(s), "position_pct": pct}


# ----------------------------------------------------------------------------------------------
# peers
# ----------------------------------------------------------------------------------------------
def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def peer_statements(ctx: dict) -> dict:
    """Peer rank, median and above/below sentences built from the peer tables (the only source for peer claims)."""
    subject = ctx["meta"]["ticker"].upper()
    out = {"sentences": [], "rank": {}}
    pe = ctx["peers"].get("ev_ebitda")
    if pe:
        vals = {c: v for c, v in pe["values"].items() if v is not None}
        if subject in {c.upper() for c in vals}:
            sv = next(v for c, v in vals.items() if c.upper() == subject)
            order = sorted(vals, key=lambda c: vals[c])
            rank = order.index(next(c for c in order if c.upper() == subject)) + 1
            med = pe["median_excl_subject"]
            rel = "below" if sv < med else ("above" if sv > med else "equal to")
            diff = abs(sv - med)
            pct = diff / med * 100 if med else 0
            nearest = min((c for c in vals if c.upper() != subject), key=lambda c: abs(vals[c] - sv))
            out["rank"]["ev_ebitda_fy_avg"] = {"rank_lowest_first": rank, "of": len(vals), "relation": rel,
                                              "diff_x": diff, "diff_pct": pct, "nearest": nearest}
            out["sentences"].append(
                f"On {pe['year']} EV/EBITDA at fiscal-year average prices, {subject} is {sv:.1f}x, the {_ordinal(rank)} lowest of {len(vals)} companies. "
                f"The peer median excluding {subject} is {med:.1f}x, so {subject} is {diff:.1f}x ({pct:.0f}%) {rel} the median. "
                f"The closest peer is {nearest} at {vals[nearest]:.1f}x.")
    pc = ctx["peers"].get("current")
    if pc:
        rows = pc["rows"]
        for col, name, med_key in (("ev_ebitda_fy", f"{ctx['meta']['latest_actual_year']} EBITDA", "median_excl_subject_fy"),
                                   ("ev_ebitda_ttm", "TTM EBITDA", "median_excl_subject_ttm")):
            vals = {t: r.get(col) for t, r in rows.items() if r.get(col) is not None}
            sv = next((v for t, v in vals.items() if t.upper() == subject), None)
            med = pc.get(med_key)
            if sv is None or med is None:
                continue
            order = sorted(vals, key=lambda t: vals[t])
            rank = order.index(next(t for t in order if t.upper() == subject)) + 1
            rel = "below" if sv < med else ("above" if sv > med else "equal to")
            out["rank"][col] = {"rank_lowest_first": rank, "of": len(vals), "relation": rel, "diff_x": abs(sv - med),
                                "diff_pct": abs(sv - med) / med * 100}
            out["sentences"].append(
                f"At current prices on {name}, {subject} trades at {sv:.1f}x, the {_ordinal(rank)} lowest of {len(vals)}. "
                f"The peer median excluding {subject} is {med:.1f}x, so {subject} is {abs(sv-med):.1f}x ({abs(sv-med)/med*100:.0f}%) {rel} the median.")
    ea = ctx["peers"].get("ebitda_actual_cr")
    if ea:
        y = max(ea)
        vals = {c: v for c, v in ea[y].items() if v is not None}
        if any(c.upper() == subject for c in vals):
            order = sorted(vals, key=lambda c: -vals[c])
            rank = order.index(next(c for c in order if c.upper() == subject)) + 1
            sv = next(v for c, v in vals.items() if c.upper() == subject)
            higher = [f"{c} (Rs {vals[c]:,.0f} Cr)" for c in order[:rank - 1]]
            out["rank"]["ebitda"] = {"rank_highest_first": rank, "of": len(vals)}
            out["sentences"].append(
                f"On {y} EBITDA, {subject} (Rs {sv:,.0f} Cr) ranks {_ordinal(rank)} of {len(vals)} by size"
                + (f"; larger: {', '.join(higher)}." if higher else "."))
    return out


# ----------------------------------------------------------------------------------------------
# TTM versus last full year
# ----------------------------------------------------------------------------------------------
def ttm_comparison(ctx: dict) -> Optional[dict]:
    ttm, fy = ctx.get("ttm"), ctx.get("latest_fy")
    if not ttm or not fy:
        return None
    rows = []

    def add(label, a, b, unit="Cr"):
        if a is None or b is None:
            return
        chg = (b / a - 1) * 100 if a else None
        rows.append({"label": label, "fy": a, "ttm": b, "change_pct": chg, "unit": unit})
    add("Revenue", fy.get("revenue_cr"), ttm.get("revenue_cr"))
    add("EBITDA", fy.get("ebitda_cr"), ttm.get("ebitda_cr"))
    add("Net income", fy.get("pat_cr"), ttm.get("pat_cr"))
    add("EPS (Rs)", fy.get("eps"), ttm.get("eps"), "Rs")
    em_fy = fy["ebitda_cr"] / fy["revenue_cr"] * 100 if fy.get("ebitda_cr") and fy.get("revenue_cr") else None
    em_t = ttm["ebitda_cr"] / ttm["revenue_cr"] * 100 if ttm.get("ebitda_cr") and ttm.get("revenue_cr") else None
    if em_fy is not None and em_t is not None:
        rows.append({"label": "EBITDA margin (%)", "fy": em_fy, "ttm": em_t, "change_pct": em_t - em_fy, "unit": "pts"})
    diverge = [r for r in rows if r["label"] in ("EBITDA", "Net income", "EPS (Rs)") and r["change_pct"] is not None and abs(r["change_pct"]) >= 25]
    text = None
    if diverge:
        parts = ", ".join(f"{r['label'].split(' (')[0]} {r['change_pct']:+.0f}%" for r in diverge)
        text = (f"Full year and TTM tell different stories: against {fy.get('label')}, trailing four quarters to {ttm['period_end']} show {parts}. "
                f"The {fy.get('label')} figures describe the downturn; the TTM figures include the recovery.")
    return {"fy_label": fy.get("label"), "ttm_period_end": ttm["period_end"], "rows": rows, "text": text}


# ----------------------------------------------------------------------------------------------
# assemble
# ----------------------------------------------------------------------------------------------
def compute_insights(ctx: dict, final_df: pd.DataFrame, financial_data: dict, extras: Optional[dict] = None) -> dict:
    """All explanatory facts, verdicts, the summary panel and fact ownership for one report."""
    extras = extras or {}
    ms = margin_series(final_df)
    ratios = financial_data.get("ratios")
    verdicts = {}

    def hist_from(s: dict) -> list:
        return list(s.values())

    def put(key, value, history, **kw):
        v = make_verdict(key, value, history, **kw)
        if v:
            verdicts[key] = v

    for key in ("contribution_margin", "ebitda_margin", "sga_pct"):
        s = ms[key]
        if s:
            put(key, s[max(s)], hist_from(s))
    growth = _row_series(final_df, "Revenue Growth")
    if growth:
        put("revenue_growth", growth[max(growth)], hist_from(growth))
    if ratios is not None and not ratios.empty:
        r = ratios.sort_values("year")
        def col(c, scale=1.0):
            return [float(x) * scale for x in r[c].tolist() if pd.notna(x) and x is not None] if c in r else []
        for key, c, sc in (("net_margin", "netProfitMargin", 100), ("roe", "returnOnEquity", 100), ("roa", "returnOnAssets", 100),
                           ("debt_equity", "debtEquityRatio", 1), ("current_ratio", "currentRatio", 1)):
            h = col(c, sc)
            if h:
                put(key, h[-1], h)
    icov = (ctx["derived"].get("interest_coverage_ebit_x") or {})
    icov_h = [v for k, v in sorted(icov.items()) if v is not None]
    if icov_h:
        put("interest_coverage", icov_h[-1], icov_h)
    cur = ctx["current_valuation"]
    pc = ctx["peers"].get("current")
    if cur.get("ev_ebitda_ttm") is not None and pc and pc.get("median_excl_subject_ttm"):
        put("ev_ebitda", cur["ev_ebitda_ttm"], [r["ev_ebitda"] for r in ctx["fy_series"].values() if r.get("ev_ebitda")],
            peer_median=pc["median_excl_subject_ttm"], label="EV/EBITDA (TTM, current price)", peer_basis="peer median at current prices")
    pe_hist = [r["pe"] for r in ctx["fy_series"].values() if r.get("pe")]
    if cur.get("pe_ttm_api") is not None and len(pe_hist) >= 3:
        put("pe", cur["pe_ttm_api"], pe_hist, label="P/E (TTM, current price)")

    # Ratio verdicts come from one source: the ratio-bar facts (peer median first, then own history, then a labelled
    # general norm). They replace the own-history-only verdicts above so the chips and the summary never disagree.
    for k, rbv in (extras.get("ratio_bars") or {}).items():
        if not isinstance(rbv, dict) or not rbv.get("verdict") or "scale" not in rbv:
            continue
        base = verdicts.get(k) or {}
        verdicts[k] = {"key": k, "label": METRIC_DEFS.get(k, (rbv.get("label", k),))[0], "unit": rbv.get("unit", base.get("unit", "")),
                       "direction": "higher_is_better" if rbv.get("higher_is_better") else "lower_is_better",
                       "verdict": rbv["verdict"], "yardstick": rbv["yardstick"], "scale": rbv["scale"],
                       "so_what": rbv["so_what"], "source_note": rbv["source_note"]}

    bridge = margin_bridge(final_df)
    oplev = operating_leverage(final_df)
    vol = volatility(final_df)
    positions = {}
    for key in ("contribution_margin", "ebitda_margin", "sga_pct"):
        s = ms[key]
        if s:
            rp = range_position(s[max(s)], list(s.values()))
            if rp:
                positions[key] = rp
    ins = {"verdicts": verdicts, "margin_bridge": bridge, "operating_leverage": oplev, "volatility": vol,
           "range_positions": positions, "peer_statements": peer_statements(ctx), "ttm_comparison": ttm_comparison(ctx)}
    ins["summary_panel"] = summary_panel(ctx, ins, extras)
    ins["fact_cards"] = fact_cards(ctx, ins)
    return ins


LIGHT = {"good": "green", "neutral": "amber", "concern": "red"}


def _word(v: Optional[dict]) -> str:
    return {"good": "good", "neutral": "neutral", "concern": "a concern"}.get(v["verdict"], "n/a") if v else "n/a"


def summary_panel(ctx: dict, ins: dict, extras: dict) -> list[dict]:
    """Four questions with traffic lights. Each answer is assembled from computed facts."""
    vd = ins["verdicts"]
    panel = []

    # 1. Is the core business healthy?
    keys = [k for k in ("contribution_margin", "interest_coverage", "revenue_growth") if k in vd]
    cf = extras.get("cashflow") or {}
    verdicts = [vd[k]["verdict"] for k in keys]
    if cf.get("answers", {}).get("profit_to_cash"):
        verdicts.append(cf["answers"]["profit_to_cash"]["verdict"])
    light = "red" if verdicts.count("concern") >= 2 else ("green" if verdicts and "concern" not in verdicts and verdicts.count("good") >= 2 else "amber")
    detail = "; ".join(f"{vd[k]['label'].lower()} is {_word(vd[k])}" for k in keys) or "not enough data for a verdict"
    panel.append({"q": "Is the core business healthy?", "light": light, "answer": detail[0].upper() + detail[1:] + ".",
                  "evidence": [vd[k]["so_what"] for k in keys[:1]]})

    # 2. Why did profit change?
    b, o = ins.get("margin_bridge"), ins.get("operating_leverage")
    light2 = LIGHT.get(vd.get("ebitda_margin", {}).get("verdict"), "amber")
    ans = " ".join(x["text"] for x in (b, o) if x) or "Not enough history for a margin bridge."
    panel.append({"q": "Why did profit change?", "light": light2, "answer": ans, "evidence": []})

    # 3. Is the price justified?
    fv, cur = ctx["fair_value"], ctx["current_valuation"]
    mid = next((g for g in fv["grid"] if g["multiple"] == 16), None) or (fv["grid"][len(fv["grid"]) // 2] if fv["grid"] else None)
    ev_v = vd.get("ev_ebitda")
    light3 = LIGHT.get(ev_v["verdict"], "amber") if ev_v else "amber"
    if mid and mid.get("vs_current_price_pct") is not None and mid["vs_current_price_pct"] < -20 and light3 == "green":
        light3 = "amber"
    parts = []
    if cur.get("ev_ebitda_ttm") is not None:
        pm = ctx["peers"].get("current", {}) or {}
        parts.append(f"At Rs {ctx['market']['price']:,.0f} (as of {ctx['market']['as_of']}) the stock trades at {cur['ev_ebitda_ttm']:.1f}x TTM EV/EBITDA"
                     + (f" against a peer median of {pm['median_excl_subject_ttm']:.1f}x at current prices." if pm.get("median_excl_subject_ttm") else "."))
    if mid:
        parts.append(f"{mid['multiple']}x our {fv['ebitda_year']} EBITDA implies Rs {mid['implied_price']:,.0f} per share, {abs(mid['vs_current_price_pct']):.0f}% "
                     f"{'below' if mid['vs_current_price_pct'] < 0 else 'above'} the price.")
    panel.append({"q": "Is the price justified?", "light": light3, "answer": " ".join(parts) or "Not enough data.", "evidence": []})

    # 4. What should I watch?
    watch = []
    if "sga_pct" in vd:
        watch.append(f"SG&A as % of revenue ({_word(vd['sga_pct'])} against its own history).")
    t = ins.get("ttm_comparison")
    if t and t.get("text"):
        watch.append("Whether TTM earnings hold; full-year and TTM figures disagree.")
    tech = extras.get("technical") or {}
    if tech.get("verdict_sentence"):
        watch.append(tech["verdict_sentence"])
    panel.append({"q": "What should I watch?", "light": "amber", "answer": " ".join(watch) or "Quarterly margins and cash conversion.", "evidence": []})
    return panel


# ----------------------------------------------------------------------------------------------
# fact ownership (state each key fact in full once)
# ----------------------------------------------------------------------------------------------
def fact_cards(ctx: dict, ins: dict) -> dict:
    cards = {}
    ms_growth = None
    lf = ctx["latest_fy"]
    if lf.get("revenue_cr") is not None:
        cards["revenue_change"] = {"owner": "investment_overview", "text": f"{lf.get('label')} revenue and its change against the prior year"}
    b = ins.get("margin_bridge")
    if b:
        cards["margin_bridge"] = {"owner": "major_takeaways", "text": b["text"]}
    o = ins.get("operating_leverage")
    if o:
        cards["operating_leverage"] = {"owner": "major_takeaways", "text": o["text"]}
    cards["valuation_multiples"] = {"owner": "valuation_overview", "text": "current EV, EV/EBITDA, P/E, fair-value grid and implied price"}
    cards["forecast_gap"] = {"owner": "valuation_overview", "text": "model forecast versus the peer-dataset trend projection, and the size of the gap"}
    cards["sga_and_margin_levels"] = {"owner": "major_takeaways", "text": "SG&A as % of revenue, contribution margin and EBITDA margin levels and their change"}
    cards["cash_conversion"] = {"owner": "investment_overview", "text": "operating cash flow, free cash flow, cash conversion and working-capital release"}
    cards["profitability_vs_peers"] = {"owner": "competitor_analysis", "text": "margins, returns, coverage and leverage against peer medians"}
    ps = ins.get("peer_statements", {}).get("sentences")
    if ps:
        cards["peer_comparison"] = {"owner": "competitor_analysis", "text": "peer ranks, medians and above/below statements"}
    t = ins.get("ttm_comparison")
    if t and t.get("text"):
        cards["ttm_vs_fy"] = {"owner": "investment_overview", "text": t["text"]}
    cards["news_events"] = {"owner": "news_summary", "text": "recent news, results announcements and price reaction"}
    cards["risks_list"] = {"owner": "risks", "text": "specific downside risks"}
    return cards


def section_guidance(ins: dict, section: str) -> str:
    """Per-section block: facts this section owns (state once, in full) and facts owned elsewhere (do not restate)."""
    cards = ins.get("fact_cards") or {}
    own = [c for c in cards.values() if c["owner"] == section]
    other = [(c["owner"], c["text"]) for c in cards.values() if c["owner"] != section]
    L = ["", "=== SECTION GUIDANCE (computed; not part of the shared facts) ===",
         f"This section is: {section}. Every key fact is stated in full in exactly one section."]
    if own:
        L.append("Facts this section owns (state them here, once):")
        L += [f"  - {c['text']}" for c in own]
    if other:
        L.append("Facts owned by other sections. Do not restate their numbers; refer to them in half a sentence at most:")
        L += [f"  - [{o}] {t}" for o, t in other]
    ps = ins.get("peer_statements", {}).get("sentences")
    if ps and section in ("competitor_analysis", "valuation_overview"):
        L.append("Peer statements generated from the peer tables (use these wording and numbers; do not make other peer claims):")
        L += [f"  - {s}" for s in ps]
    vd = ins.get("verdicts") or {}
    if vd and section in ("investment_overview", "major_takeaways", "valuation_overview", "competitor_analysis"):
        L.append("Verdicts with their yardstick (use the verdict word and the so-what, do not invent others):")
        for v in vd.values():
            L.append(f"  - {v['label']}: {v['verdict']} against {v['yardstick'].replace('_', ' ')}"
                     f"{'' if v['yardstick'] != 'general_norm' else ' (' + GENERAL_NORM_LABEL + ')'}. So what: {v['so_what']}")
    t = ins.get("ttm_comparison")
    if t and t.get("text") and section in ("investment_overview", "valuation_overview", "news_summary"):
        L.append(f"Full year versus TTM: {t['text']}")
    if section in ("major_takeaways", "investment_overview"):
        for k in ("margin_bridge", "operating_leverage", "volatility"):
            x = ins.get(k)
            if x and x.get("text"):
                L.append(f"{k.replace('_', ' ').capitalize()}: {x['text']}")
    L.append("=== END SECTION GUIDANCE ===")
    return "\n".join(L)


def render_insights_block(ins: dict) -> str:
    """Compact text of the explanatory facts for the shared facts block (identical for every section)."""
    L = ["", "EXPLANATORY FACTS (computed in code):"]
    for k in ("margin_bridge", "operating_leverage", "volatility"):
        x = ins.get(k)
        if x and x.get("text"):
            L.append(f"  {k.replace('_', ' ')}: {x['text']}")
    t = ins.get("ttm_comparison")
    if t:
        L.append("  FY vs TTM: " + "; ".join(
            f"{r['label']} {r['fy']:,.{2 if r['unit'] == 'Rs' else 1}f} -> {r['ttm']:,.{2 if r['unit'] == 'Rs' else 1}f}" + (f" ({r['change_pct']:+.0f}%)" if r['unit'] != 'pts' and r['change_pct'] is not None else
                                                                   (f" ({r['change_pct']:+.1f} pts)" if r['change_pct'] is not None else ""))
            for r in t["rows"]))
        if t.get("text"):
            L.append(f"  {t['text']}")
    for k, rp in (ins.get("range_positions") or {}).items():
        L.append(f"  {METRIC_DEFS[k][0]} sits at {rp['position_pct']:.0f}% of its own five-year range ({rp['min']:.1f} to {rp['max']:.1f}).")
    return "\n".join(L)
