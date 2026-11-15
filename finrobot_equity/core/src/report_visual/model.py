#!/usr/bin/env python
# coding: utf-8
"""
Report model builder.

``build_report_model(analysis_dir)`` reads the validated analysis outputs (``report_context.json`` and the CSVs), runs the
verdict engine, computes fair-value cases, signposts, lights and headlines, asks ``charts`` for the chart specs, validates
the result and returns one JSON-serialisable dict (schema: SCHEMA.md). Both renderers read only that dict.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import statistics
from typing import Optional

import pandas as pd

from . import headlines as hl
from . import verdict_engine as ve

CRORE = 1e7
SCHEMA_VERSION = 1
PANEL_ORDER = [
    ("summary", "Summary", "Should I worry, and about what?"),
    ("profit_engine", "Profit", "Where does the profit come from, and where did it go?"),
    ("cyclicality", "Cycles", "How steady is the business through the cycle?"),
    ("strength", "Strength", "Can it fund itself and carry its debt?"),
    ("valuation", "Valuation", "Is the price justified?"),
    ("price", "Price", "What is the share price telling us?"),
    ("watch", "Watch", "What should I watch next?"),
]
DISCLAIMER = ("This report is for information only and is not investment advice. Figures come from the data provider named in "
              "the sources section and from calculations shown in the data notes. Forecasts are estimates, not promises.")


class ReportModelError(RuntimeError):
    pass


def fy(label: Optional[str]) -> Optional[str]:
    """'2026A' -> 'FY26'; '2027E' -> 'FY27E'; others unchanged."""
    if not label:
        return label
    m = re.fullmatch(r"(\d{4})([AE])", str(label))
    return f"FY{m.group(1)[2:]}{'E' if m.group(2) == 'E' else ''}" if m else str(label)


def _f(x) -> Optional[float]:
    try:
        v = float(str(x).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _cv(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 3 or not statistics.mean(xs):
        return None
    return statistics.stdev(xs) / abs(statistics.mean(xs)) * 100


# ----------------------------------------------------------------------------------------------
# facts
# ----------------------------------------------------------------------------------------------
def compute_facts(ctx: dict, final_df: pd.DataFrame) -> dict:
    from modules import insights as ins
    ms = ins.margin_series(final_df)
    growth = ins._row_series(final_df, "Revenue Growth")
    cm, sga, em = ms["contribution_margin"], ms["sga_pct"], ms["ebitda_margin"]
    years = sorted(set(cm) & set(sga) & set(em))
    latest = years[-1] if years else None
    f: dict = {"years": years, "latest_year": latest, "latest_label": fy(latest)}
    if growth:
        f["growth_range"] = max(growth.values()) - min(growth.values())
        f["growth_observations"] = len(growth)
        downs = [y for y, g in growth.items() if g < 0]
        f["downturn_years"] = len(downs)
        hit = True
        for y in downs:
            prev = [k for k in years if k < y]
            if prev and em.get(y) is not None and em.get(prev[-1]) is not None:
                hit = hit and em[y] < em[prev[-1]]
        f["downturns_hit_margin"] = bool(downs) and hit
    if cm:
        f["cm_range"] = max(cm.values()) - min(cm.values())
    if sga and len(years) >= 2:
        f["sga_change"] = sga[years[-1]] - sga[years[0]]
    f["avg_ebitda_margin"] = statistics.mean(em.values()) if em else None
    ol = (ctx.get("insights") or {}).get("operating_leverage") or {}
    f["per100"] = abs(ol["ebitda_per_100_revenue"]) if ol.get("ebitda_per_100_revenue") is not None else None
    cvv, cur = ctx["current_valuation"], ctx["market"]
    f["price"], f["shares_mn"] = cur.get("price"), cur.get("shares_mn")
    f["net_debt_cr"], f["ev_cr"], f["eps_ttm"] = cvv.get("net_debt_cr"), cvv.get("ev_cr"), cvv.get("eps_ttm")
    f["pe_ttm"] = cvv.get("pe_ttm_api")
    # balance sheet trend
    de_hist = [v for _, v in sorted(((ctx.get("extras") or {}).get("ratio_bars", {}).get("debt_equity", {}).get("history") or {}).items())]
    if len(de_hist) >= 2:
        a, b = de_hist[0], de_hist[-1]
        f["debt_equity_trend"] = "falling" if b < a * 0.9 else ("rising" if b > a * 1.1 else "steady")
        f["debt_equity_first"], f["debt_equity_last"] = a, b
    cf = (ctx.get("extras") or {}).get("cashflow") or {}
    if cf.get("per_year"):
        lab = cf["years"][-1] if cf.get("years") else None
        f["conversion_latest_pct"] = (cf["per_year"][lab]["conversion"] or 0) * 100 if lab and cf["per_year"][lab].get("conversion") is not None else None
        f["cash_cyclical_flag"] = bool((cf.get("cyclicality") or {}).get("flag"))
        agg = (cf.get("aggregate") or {}).get("conversion")
        f["conversion_aggregate_pct"] = agg * 100 if agg is not None else None
    # valuation relative positions
    ev_hist = [r["ev_ebitda"] for r in ctx["fy_series"].values() if r.get("ev_ebitda")]
    now = cvv.get("ev_ebitda_ttm")
    if now and ev_hist:
        f["ev_vs_history"] = "above" if now > max(ev_hist) else ("below" if now < min(ev_hist) else "within")
    pc = ctx["peers"].get("current") or {}
    med = pc.get("median_excl_subject_ttm")
    if now and med:
        f["ev_vs_peers"] = "above" if now > 1.10 * med else ("below" if now < 0.90 * med else "within")
    # steadiest peer and EBITDA volatility
    ea = ctx["peers"].get("ebitda_actual_cr") or {}
    subj = ctx["meta"]["ticker"].upper()
    series: dict[str, list] = {}
    for _y, row in sorted(ea.items()):
        for t, v in row.items():
            series.setdefault(t.upper(), []).append(v)
    cvs = {t: _cv(v) for t, v in series.items()}
    peer_cv = {t: c for t, c in cvs.items() if t != subj and c is not None}
    f["ebitda_cv"] = cvs.get(subj)
    f["peer_ebitda_cv_median"] = _median(list(peer_cv.values()))
    if peer_cv:
        steady = min(peer_cv, key=peer_cv.get)
        f["steadiest_peer"] = steady
        f["steadiest_peer_ebitda_band"] = (min(series[steady]), max(series[steady]))
        st_mult = (pc.get("rows", {}).get(steady) or {}).get("ev_ebitda_ttm")
        f["steadiest_peer_below_subject"] = bool(st_mult and now and st_mult < now)
    ttm, lf = ctx.get("ttm") or {}, ctx.get("latest_fy") or {}
    f["recovery_narrative"] = bool(ttm.get("ebitda_cr") and lf.get("ebitda_cr") and ttm["ebitda_cr"] >= 1.25 * lf["ebitda_cr"]
                                   and (_median(list(em.values())) or 0) > (em.get(latest) or 0))
    return f


# ----------------------------------------------------------------------------------------------
# fair value cases
# ----------------------------------------------------------------------------------------------
def fair_value_cases(ctx: dict, facts: dict) -> list[dict]:
    price, shares, nd = facts.get("price"), facts.get("shares_mn"), facts.get("net_debt_cr")
    if not (price and shares) or nd is None:
        return []
    hist = [r["ev_ebitda"] for r in ctx["fy_series"].values() if r.get("ev_ebitda")]
    pe, pc = ctx["peers"].get("ev_ebitda") or {}, ctx["peers"].get("current") or {}
    meds = [m for m in (pe.get("median_excl_subject"), pc.get("median_excl_subject_ttm")) if m]
    fv = ctx["fair_value"]
    tp = ctx["peers"].get("trend_projection") or {}
    subj = ctx["meta"]["ticker"].upper()
    proj = next((v for t, v in (tp.get("ebitda_cr") or {}).items() if t.upper() == subj), None)
    ttm_e = (ctx.get("ttm") or {}).get("ebitda_cr")

    def case(key, label, basis, ebitda, lo_m, hi_m):
        lo = (lo_m * ebitda - nd) * 10 / shares
        hi = (hi_m * ebitda - nd) * 10 / shares
        lo, hi = min(lo, hi), max(lo, hi)
        return {"key": key, "label": label, "basis": basis, "ebitda_cr": ebitda, "low_mult": min(lo_m, hi_m), "high_mult": max(lo_m, hi_m),
                "low_price": lo, "high_price": hi, "vs_price_pct_low": (lo / price - 1) * 100, "vs_price_pct_high": (hi / price - 1) * 100}
    out = []
    if fv.get("ebitda_cr") and hist:
        out.append(case("model_own_history", "Company model at own-history multiples",
                        f"{fy(fv['ebitda_year'])} model EBITDA x own EV/EBITDA range", fv["ebitda_cr"], min(hist), max(hist)))
    if fv.get("ebitda_cr") and meds:
        out.append(case("model_peer_median", "Company model at the peer median",
                        f"{fy(fv['ebitda_year'])} model EBITDA x peer median EV/EBITDA", fv["ebitda_cr"], min(meds), max(meds)))
    if proj and meds:
        out.append(case("peer_projection_peer_median", "Peer-dataset projection at the peer median",
                        f"FY{str(tp.get('year'))[2:]} trend projection x peer median", proj, min(meds), max(meds)))
    if ttm_e and meds:
        out.append(case("ttm_peer_median", "Latest four quarters at the peer median", "TTM EBITDA x peer median", ttm_e, min(meds), max(meds)))
    return out


# ----------------------------------------------------------------------------------------------
# signposts
# ----------------------------------------------------------------------------------------------
def _status(cur: float, thr: float, higher_better: bool, near: float = 0.15) -> str:
    if higher_better:
        return "good" if cur >= thr else ("neutral" if cur >= thr * (1 - near) else "concern")
    return "good" if cur <= thr else ("neutral" if cur <= thr * (1 + near) else "concern")


def build_signposts(ctx: dict, final_df: pd.DataFrame, facts: dict) -> list[dict]:
    from modules import insights as ins
    ms = ins.margin_series(final_df)
    out = []
    sga = ms["sga_pct"]
    if sga:
        cur = sga[max(sga)]
        thr = float(round(statistics.median(sga.values())))
        out.append({"metric": "SG&A as % of revenue", "current": cur, "current_text": f"{cur:.1f}% ({facts['latest_label']})",
                    "threshold_text": f"Below {thr:.0f}% of revenue", "status": _status(cur, thr, False),
                    "why": "Overheads above their own five-year median are what turned a sales dip into a profit fall."})
    ttm = ctx.get("ttm") or {}
    em = ms["ebitda_margin"]
    if ttm.get("ebitda_cr") and ttm.get("revenue_cr") and em:
        cur = ttm["ebitda_cr"] / ttm["revenue_cr"] * 100
        thr = float(round(statistics.median(em.values())))
        out.append({"metric": "EBITDA margin", "current": cur, "current_text": f"{cur:.1f}% (TTM)",
                    "threshold_text": f"Above {thr:.0f}% for two to three quarters", "status": _status(cur, thr, True, 0.15),
                    "why": "A margin back at its own median for several quarters would show the recovery is not a one-quarter event."})
    agg = facts.get("conversion_aggregate_pct")
    lat = facts.get("conversion_latest_pct")
    if agg and lat is not None:
        thr = float(round(agg))
        flattered = facts.get("cash_cyclical_flag") and lat >= 1.2 * agg
        out.append({"metric": "Cash conversion", "current": lat, "current_text": f"{lat:.0f}% ({facts['latest_label']}{', flattered' if flattered else ''})",
                    "threshold_text": f"Near {thr:.0f}% while sales grow", "status": "concern" if flattered else _status(lat, thr, True, 0.2),
                    "why": "Cash that stays near its long-run level as sales recover would show profit is real, not released working capital."})
    ic = ((ctx.get("extras") or {}).get("ratio_bars") or {}).get("interest_coverage") or {}
    if ic.get("scale") and ic.get("history") and ic["scale"]["value"] < 50:      # a near debt-free company has no coverage signpost
        thr = float(round(statistics.median(ic["history"].values())))
        cur = ic["scale"]["value"]
        out.append({"metric": "Interest coverage", "current": cur, "current_text": f"{cur:.1f}x", "threshold_text": f"Above {thr:.0f}x",
                    "status": _status(cur, thr, True, 0.2), "why": "Coverage back above its own median would take pressure off debt service."})
    cv, pc = ctx["current_valuation"], ctx["peers"].get("current") or {}
    if cv.get("ev_ebitda_ttm") and pc.get("median_excl_subject_ttm"):
        cur, thr = cv["ev_ebitda_ttm"], pc["median_excl_subject_ttm"]
        out.append({"metric": "EV/EBITDA (TTM)", "current": cur, "current_text": f"{cur:.1f}x", "threshold_text": f"At or below {thr:.1f}x (peer median)",
                    "status": _status(cur, thr, False, 0.10), "why": "Earnings growth, not a falling share price, is how the multiple should close the gap to peers."})
    for sp in out:
        m = re.search(r"(\d+(?:\.\d+)?)", sp["threshold_text"])
        sp["threshold"] = float(m.group(1)) if m else None
    return out[:5]


# ----------------------------------------------------------------------------------------------
# chips, lights, panels
# ----------------------------------------------------------------------------------------------
def build_chips(ctx: dict, final_df: pd.DataFrame, facts: dict, cases: list[dict]) -> dict[str, dict]:
    from modules import insights as ins
    ms = ins.margin_series(final_df)
    growth = ins._row_series(final_df, "Revenue Growth")
    rb = (ctx.get("extras") or {}).get("ratio_bars") or {}
    per = facts["latest_label"]
    chips: dict[str, dict] = {}

    def put(c):
        if c:
            chips[c["id"]] = c

    def rbf(key):
        r = rb.get(key) or {}
        return (r.get("scale") or {}), [v for _, v in sorted((r.get("history") or {}).items())]
    for key in ("contribution_margin", "sga_pct", "ebitda_margin"):
        s = ms[key]
        if not s:
            continue
        peer = rbf(key)[0].get("peer_median") if key != "contribution_margin" else None
        put(ve.evaluate(key, s[max(s)], list(s.values()), peer, facts, per))
    put(ve.operating_leverage(facts.get("per100"), facts.get("avg_ebitda_margin"), facts, f"{fy(facts['years'][-2])} to {per}" if len(facts["years"]) > 1 else per))
    if growth:
        put(ve.evaluate("revenue_growth", growth[max(growth)], list(growth.values()), None, facts, per))
    if facts.get("ebitda_cv") is not None and facts.get("peer_ebitda_cv_median") is not None:
        put(ve.evaluate("ebitda_volatility", facts["ebitda_cv"], [], facts["peer_ebitda_cv_median"], facts, "5 years",
                        scale_override={"min": 0.0, "max": max(facts["ebitda_cv"], facts["peer_ebitda_cv_median"]) * 1.2}))
    cf = (ctx.get("extras") or {}).get("cashflow") or {}
    if cf.get("per_year") and facts.get("conversion_latest_pct") is not None:
        hist = [(cf["per_year"][y].get("conversion") or 0) * 100 for y in cf["years"]]
        put(ve.cash_conversion(facts["conversion_latest_pct"], hist, facts.get("conversion_aggregate_pct"), facts.get("cash_cyclical_flag", False), per))
    for key in ("debt_equity", "interest_coverage", "roe", "net_margin", "roa", "current_ratio", "gross_margin"):
        sc, hist = rbf(key)
        if sc.get("value") is not None:
            put(ve.evaluate(key, sc["value"], hist, sc.get("peer_median"), facts, per))
    cvv, pc = ctx["current_valuation"], ctx["peers"].get("current") or {}
    pe_hist = [r["pe"] for r in ctx["fy_series"].values() if r.get("pe")]
    if cvv.get("pe_ttm_api"):
        put(ve.evaluate("pe", cvv["pe_ttm_api"], pe_hist, None, facts, "TTM"))
    ev_hist = [r["ev_ebitda"] for r in ctx["fy_series"].values() if r.get("ev_ebitda")]
    if cvv.get("ev_ebitda_ttm"):
        put(ve.evaluate("ev_ebitda", cvv["ev_ebitda_ttm"], ev_hist, pc.get("median_excl_subject_ttm"), facts, "TTM"))
    if cases and facts.get("price"):
        put(ve.fair_value(facts["price"], [(c["low_price"], c["high_price"]) for c in cases], "snapshot"))
    return chips


def build_tests(ctx: dict) -> list[dict]:
    cfg = ve.load_config()["technical_tests"]
    tech = (ctx.get("extras") or {}).get("technical") or {}
    tests = []
    for t in tech.get("tests") or []:
        key = t["test"].lower()
        rules = cfg.get(key, {"default": "neutral"})
        state = t["state"].lower()
        verdict = rules.get("default", "neutral")
        for v in ("concern", "neutral", "good"):          # order matters: "Up, fading" must hit the fading rule before "up"
            if any(k in state for k in rules.get(v, [])) and (v == "neutral" or True):
                if v == "good" and any(k in state for k in rules.get("neutral", [])):
                    continue
                verdict = v
                break
        tests.append({"key": key, "label": t["test"], "measures": t["measures"], "state": t["state"], "verdict": verdict,
                      "icon": ve.ICON[verdict], "evidence": t["evidence"], "so_what": t["evidence"]})
    return tests


def build_lights(chips: dict, tests: list[dict], facts: dict) -> list[dict]:
    def vs(*keys):
        return [chips[k]["verdict"] for k in keys if k in chips]
    core = vs("contribution_margin", "debt_equity", "interest_coverage")
    biz = "concern" if (chips.get("contribution_margin", {}).get("verdict") == "concern" or core.count("concern") >= 2) else (
        "good" if core and "concern" not in core else "neutral")
    prof_v = vs("ebitda_margin", "sga_pct")
    profit = "concern" if "concern" in prof_v[:1] else ("neutral" if "concern" in prof_v else "good")
    price = (chips.get("fair_value") or chips.get("ev_ebitda") or {}).get("verdict", "neutral")
    tv = {t["key"]: t["verdict"] for t in tests}
    mom = ("concern" if tv.get("trend") == "concern" else
           "good" if tv.get("trend") == "good" and tv.get("momentum") == "good" and tv.get("stretch") != "concern" else "neutral")
    lat = facts["latest_label"]
    r = {
        "business": f"Pricing {chips['contribution_margin']['verdict'] if 'contribution_margin' in chips else 'n/a'}; debt {chips.get('debt_equity', {}).get('verdict', 'n/a')}; interest cover {chips.get('interest_coverage', {}).get('verdict', 'n/a')}.",
        "profit": f"EBITDA margin {chips.get('ebitda_margin', {}).get('verdict', 'n/a')} and overheads {chips.get('sga_pct', {}).get('verdict', 'n/a')} for {lat}.",
        "price": f"Fair value {chips.get('fair_value', {}).get('verdict', 'n/a')} and EV/EBITDA {chips.get('ev_ebitda', {}).get('verdict', 'n/a')} at the snapshot price.",
        "momentum": "; ".join(f"{t['label'].lower()} {t['verdict']}" for t in tests) + "." if tests else "Price history not available.",
    }
    r = {k: v.replace("concern", "a concern") for k, v in r.items()}
    names = {"business": "Business health", "profit": "Profit trend", "price": "Price vs value", "momentum": "Momentum"}
    lights = {"business": biz, "profit": profit, "price": price, "momentum": mom}
    return [{"key": k, "label": names[k], "light": lights[k], "icon": ve.ICON[lights[k]], "reason": r[k]} for k in names]


def build_warnings(ctx: dict, facts: dict, cases: list[dict]) -> list[dict]:
    w = []
    for c in ctx.get("checks") or []:
        if c.get("ok") is False:
            w.append({"level": "error", "text": f"Reconciliation failed: {c['name']}."})
    cf = (ctx.get("extras") or {}).get("cashflow") or {}
    if cf and cf.get("capex_available") is False:
        w.append({"level": "warning", "text": "Capex cannot be separated from investing cash flow in this data, so free cash flow is not shown."})
    ttm = ctx.get("ttm")
    if ttm:
        w.append({"level": "info", "text": f"The quarter to {ttm['period_end']} is in the TTM figures but not in the full-year tables."})
    else:
        w.append({"level": "warning", "text": "TTM figures are unavailable (four consecutive quarters were not found)."})
    w.append({"level": "info", "text": "No analyst consensus is available from this data source. Fair-value cases use the company model, "
                                      "the peer-dataset projection and peer multiples."})
    if not ctx["peers"].get("current"):
        w.append({"level": "warning", "text": "Peer multiples at current prices are unavailable."})
    for u in (ctx.get("derived") or {}).get("unavailable", []):
        w.append({"level": "info", "text": f"Not available: {u}."})
    return w


def build_report_model(analysis_dir: str, report_date: Optional[str] = None, kind: Optional[str] = None,
                       charts_fn=None) -> dict:
    from modules import report_context as rc
    from . import validate as vmod
    ctx = rc.load_context(os.path.join(analysis_dir, "report_context.json"))
    if not ctx:
        raise ReportModelError(f"No report_context.json in {analysis_dir}. Run the analysis step first "
                               "(bank reports are not supported by the visual formats yet).")
    kind = kind or ("bank" if ctx.get("kind") == "bank" else "nonfinancial")
    if kind == "bank":
        from .bank.build import build_bank_report_model
        return build_bank_report_model(analysis_dir, report_date=report_date, charts_fn=charts_fn)
    if kind != "nonfinancial":
        raise ReportModelError(f"Unknown report kind: {kind}")
    final_df = pd.read_csv(os.path.join(analysis_dir, "financial_metrics_and_forecasts.csv"))
    facts = compute_facts(ctx, final_df)
    cases = fair_value_cases(ctx, facts)
    chips = build_chips(ctx, final_df, facts, cases)
    tests = build_tests(ctx)
    lights = build_lights(chips, tests, facts)
    light_map = {l["key"]: l["light"] for l in lights}
    test_map = {t["key"]: t["verdict"] for t in tests}
    signposts = build_signposts(ctx, final_df, facts)
    warnings = build_warnings(ctx, facts, cases)

    if charts_fn is None:
        try:
            from .charts import build_panel_charts as charts_fn
        except ImportError:
            charts_fn = None
    charts: dict = {}
    if charts_fn:
        try:
            charts = charts_fn(ctx, analysis_dir, kind) or {}
        except Exception as e:                       # charts must never block the model; the gap is shown on Panel 1
            warnings.append({"level": "warning", "text": f"Charts could not be built: {e}"})
    else:
        warnings.append({"level": "warning", "text": "Chart specifications are not available in this build."})

    tech = (ctx.get("extras") or {}).get("technical") or {}
    pve = tech.get("price_vs_earnings") or {}
    m = ctx["market"]
    meta = {"ticker": ctx["meta"]["ticker"], "company_name": ctx["meta"]["company_name"], "sector": ctx["meta"].get("sector"),
            "kind": kind, "report_date": report_date or ctx["meta"]["as_of"], "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "currency": "INR",
            "snapshot": {"price": m["price"], "price_date": m.get("price_date"), "price_time": None, "market_cap_cr": m["market_cap_cr"],
                         "week52_low": m.get("week52_low"), "week52_high": m.get("week52_high"), "shares_mn": m.get("shares_mn"),
                         "net_debt_cr": facts.get("net_debt_cr"), "ev_cr": facts.get("ev_cr"), "eps_ttm": facts.get("eps_ttm")}}

    def pick(*keys):
        return [chips[k] for k in keys if k in chips]
    n_sign = len(signposts)
    panels = []
    for n, (pid, nav, question) in enumerate(PANEL_ORDER, 1):
        p = {"id": pid, "n": n, "nav": nav, "question": question, "charts": charts.get(pid, []), "chips": [], "extra": {}, "notes": []}
        if pid == "summary":
            p["headline"] = hl.summary(light_map)
            p["extra"] = {"lights": lights, "price_vs_earnings": pve or None,
                          "header_strip": {"company": meta["company_name"], "ticker": meta["ticker"], **meta["snapshot"]}}
        elif pid == "profit_engine":
            p["headline"] = hl.profit_engine(chips, facts)
            p["chips"] = pick("contribution_margin", "sga_pct", "operating_leverage")
            p["extra"] = {"bridge": (ctx["insights"] or {}).get("margin_bridge"), "operating_leverage": (ctx["insights"] or {}).get("operating_leverage")}
        elif pid == "cyclicality":
            p["headline"] = hl.cyclicality(chips, facts)
            p["chips"] = pick("ebitda_margin", "ebitda_volatility")
        elif pid == "strength":
            p["headline"] = hl.strength(chips, facts)
            p["chips"] = pick("cash_conversion", "debt_equity", "interest_coverage")
            cf = (ctx.get("extras") or {}).get("cashflow") or {}
            if cf.get("capex_available") is False:
                p["notes"].append("Capex cannot be separated from investing cash flow in this data, so it is not shown as capex.")
            elif cf.get("capex_basis"):
                p["notes"].append(cf["capex_basis"])
        elif pid == "valuation":
            p["headline"] = hl.valuation(chips, facts)
            p["chips"] = pick("pe", "ev_ebitda", "fair_value")
            avg = next((r for k, r in sorted(ctx["fy_series"].items()) if k == facts["latest_year"]), None) or {}
            p["extra"] = {"fair_value_cases": cases, "price": facts.get("price"),
                          "toggle_note": (f"At the report's earlier price of about Rs {avg['avg_price']:.0f} (the fiscal-year average), EV/EBITDA was "
                                          f"{avg['ev_ebitda']:.1f}x.") if avg.get("avg_price") and avg.get("ev_ebitda") else None}
        elif pid == "price":
            p["headline"] = hl.price_behaviour(test_map, facts)
            p["extra"] = {"tests": tests, "verdict_sentence": tech.get("verdict_sentence"), "divergence": tech.get("divergence"),
                          "divergence_sentence": tech.get("divergence_sentence"), "rsi_zone": tech.get("rsi_zone"),
                          "readings": _readings(tech)}
        elif pid == "watch":
            p["headline"] = hl.watch(n_sign, facts)
            p["extra"] = {"signposts": signposts, "portfolio": {"current_weight": None, "target_weight": None, "triggers": None}}
        panels.append(p)

    used = {c["id"] for p in panels for c in p["chips"]}
    gl_terms = ve.load_glossary()
    model = {
        "schema_version": SCHEMA_VERSION, "meta": meta, "thesis": hl.thesis(chips, facts, light_map), "warnings": warnings, "panels": panels,
        "glossary": gl_terms,
        "sources": {"provider": "FinEdge API (NSE/BSE company filings and quotes)",
                    "snapshot": f"Price {m['price']} on {m.get('price_date')}; report date {meta['report_date']}.",
                    "definitions": [f"{c['label']}: {c['calc']}" for p in panels for c in p["chips"]],
                    "validation": {"errors": [], "warnings": [], "passed": []}},
        "disclaimer": DISCLAIMER, "facts": {k: v for k, v in facts.items() if not isinstance(v, (dict,)) and k not in ("years",)},
        "headline_rules": hl.HEADLINE_RULES,
    }
    model["facts"]["years"] = facts.get("years")
    result = vmod.validate_model(model)
    model["sources"]["validation"] = result
    for e in result["errors"]:
        model["warnings"].insert(0, {"level": "error", "text": e})
    for w in result["warnings"]:
        model["warnings"].append({"level": "warning", "text": w})
    return model


def _readings(tech: dict) -> list:
    try:
        from modules import technical_analysis as ta
        return [list(r) for r in ta.readings_table_rows(tech)]
    except Exception:
        return []


def save_report_model(model: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model['meta']['ticker']}_{model['meta']['report_date']}.report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1, ensure_ascii=False, default=str)
    return path
