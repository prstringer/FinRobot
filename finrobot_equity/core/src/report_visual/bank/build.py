#!/usr/bin/env python
# coding: utf-8
"""
Bank report model: the same seven panels and the same schema as the non-financial model, assembled from
``variants/bank.yaml``. Entry point: ``build_bank_report_model(analysis_dir, report_date=None, charts_fn=None)``.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Optional

from .. import model as M
from .. import validate as vmod
from .. import verdict_engine as ve
from . import headlines as bh
from . import providers as P

# glossary terms of the non-financial report that do not apply to banks
_DROP_TERMS = {"contribution_margin", "sga", "ebitda", "operating_leverage", "margin_bridge", "cash_conversion", "working_capital", "capex", "fcf",
               "interest_coverage", "debt_equity", "ev", "ev_ebitda", "net_debt"}


def _warnings(ctx: dict, facts: dict) -> list[dict]:
    w = []
    for t in ctx.get("data_warnings") or []:
        w.append({"level": "warning" if "like-for-like" in t else "info", "text": t})
    ttm = ctx.get("ttm")
    if ttm:
        w.append({"level": "info", "text": f"The quarter to {ttm['period_end']} is in the TTM figures but not in the full-year tables."})
    else:
        w.append({"level": "warning", "text": "TTM earnings are unavailable (four consecutive quarters were not found)."})
    w.append({"level": "info", "text": "No analyst consensus is available from this data source. Fair-value cases use the bank's own P/B and P/E history, "
                                      "the peer range and a simple EPS projection (last EPS grown at the revenue growth assumption)."})
    if not (ctx["peers"].get("bank") or {}).get("median_excl_subject", {}).get("pb_now"):
        w.append({"level": "warning", "text": "Peer P/B at today's prices is unavailable; the peer comparison uses fiscal-year-average prices."})
    for u in (ctx.get("derived") or {}).get("unavailable", []):
        w.append({"level": "info", "text": f"Not available: {u}."})
    return w


def _bridge(ctx: dict, chips: dict, facts: dict) -> Optional[dict]:
    nim, ci = P.actual_series(ctx, "NIM"), P.actual_series(ctx, "Cost-to-Income")
    ys = [y for y in nim if y in ci]
    if len(ys) < 2:
        return None
    a, b = ys[0], ys[-1]
    d_nim, d_ci = nim[b] - nim[a], ci[b] - ci[a]
    cc = (chips.get("credit_cost") or {}).get("value")
    text = (f"From {P.fy(a)} to {P.fy(b)}, the net interest margin moved {d_nim:+.2f} points and cost to income {d_ci:+.1f} points"
            + (f"; credit cost was {cc:.2f}% in {P.fy(b)}." if cc is not None else "."))
    return {"from": P.fy(a), "to": P.fy(b), "d_nim_pts": d_nim, "d_cost_to_income_pts": d_ci, "text": text}


def build_bank_report_model(analysis_dir: str, report_date: Optional[str] = None, charts_fn=None) -> dict:
    from modules import report_context as rc
    ctx = rc.load_context(os.path.join(analysis_dir, "report_context.json"))
    if not ctx or (ctx.get("kind") != "bank" and ctx["meta"].get("kind") != "bank"):
        raise M.ReportModelError(f"No bank report_context.json in {analysis_dir}. Run the analysis step for a bank first.")
    variant = P.load_variant()
    facts = P.compute_facts(ctx, variant)
    cases = P.fair_value_cases(ctx, variant, facts)
    chips = P.build_chips(ctx, variant, facts, cases)
    if "fair_value" in chips:
        chips["fair_value"]["calc"] = ("Fair value per share = multiple x book value per share or EPS, for each case, against the snapshot price. "
                                       "Cases: " + "; ".join(c["label"] for c in cases) + ".")
    tests = M.build_tests(ctx)
    lights = P.build_lights(variant, chips, tests)
    light_map = {l["key"]: l["light"] for l in lights}
    test_map = {t["key"]: t["verdict"] for t in tests}
    signposts = P.build_signposts(ctx, variant, facts)
    warnings = _warnings(ctx, facts)

    charts: dict = {}
    try:
        if charts_fn is not None:
            charts = charts_fn(ctx, analysis_dir, "bank") or {}
        else:
            from . import charts as bc
            charts = bc.build_bank_panel_charts(ctx, analysis_dir, variant, cases, facts.get("price"))
            for e in bc.LAST_ERRORS:
                if "skipped" not in e:
                    warnings.append({"level": "warning", "text": f"Chart not built: {e}"})
    except Exception as e:
        warnings.append({"level": "warning", "text": f"Charts could not be built: {e}"})

    tech = (ctx.get("extras") or {}).get("technical") or {}
    pve = tech.get("price_vs_earnings") or {}
    m = ctx["market"]
    meta = {"ticker": ctx["meta"]["ticker"], "company_name": ctx["meta"]["company_name"], "sector": ctx["meta"].get("sector"), "kind": "bank",
            "report_date": report_date or ctx["meta"]["as_of"], "generated_at": _dt.datetime.now().isoformat(timespec="seconds"), "currency": "INR",
            "snapshot": {"price": m["price"], "price_date": m.get("price_date"), "price_time": None, "market_cap_cr": m["market_cap_cr"],
                         "week52_low": m.get("week52_low"), "week52_high": m.get("week52_high"), "shares_mn": m.get("shares_mn"),
                         "net_debt_cr": None, "ev_cr": None, "eps_ttm": facts.get("eps_ttm")}}

    bridge = _bridge(ctx, chips, facts)
    panels = []
    for n, (pid, nav, _q) in enumerate(M.PANEL_ORDER, 1):
        spec = variant["panels"][pid]
        p = {"id": pid, "n": n, "nav": nav, "question": spec["question"], "charts": charts.get(pid, []),
             "chips": [chips[c] for c in spec.get("chips", []) if c in chips], "extra": {}, "notes": []}
        p["headline"] = bh.headline(spec["headline_rule"], chips, facts, lights=light_map, tests=test_map, n_signposts=len(signposts))
        if pid == "summary":
            p["extra"] = {"lights": lights, "price_vs_earnings": pve or None, "header_strip": {"company": meta["company_name"], "ticker": meta["ticker"], **meta["snapshot"]}}
        elif pid == "profit_engine":
            p["extra"] = {"bridge": None, "operating_leverage": None, "bank_bridge": bridge}
            if bridge:
                p["notes"].append(bridge["text"])
        elif pid == "cyclicality":
            if facts.get("merger_years"):
                p["notes"].append(f"Loan growth in {', '.join(facts['merger_years'])} is merger-affected and left out of the range used for the verdict.")
        elif pid == "strength":
            p["extra"] = {"also_tracked": [chips[c] for c in ("nnpa", "bank_roa", "credit_cost") if c in chips]}
        elif pid == "valuation":
            sub = meta["ticker"]
            latest = ctx["meta"]["latest_actual_year"]
            avg = ctx["fy_series"].get(latest) or {}
            roe_chip = chips.get("bank_roe")
            roe_view = None
            if roe_chip and chips.get("pb") and roe_chip["scale"].get("peer_median") is not None and facts.get("peer_pb_median"):
                roe_view = (f"{sub} earns a {roe_chip['value']:.1f}% return on equity against a peer median of {roe_chip['scale']['peer_median']:.1f}% "
                            f"and trades at {chips['pb']['value']:.2f}x book against a peer median of {facts['peer_pb_median']:.2f}x.")
            p["extra"] = {"fair_value_cases": cases, "price": facts.get("price"), "roe_view": roe_view,
                          "toggle_note": (f"At the report's earlier price of about Rs {avg['avg_price']:.0f} (the fiscal-year average), P/B was {avg['pb']:.2f}x.")
                          if avg.get("avg_price") and avg.get("pb") else None}
            if roe_view:
                p["notes"].append(roe_view)
        elif pid == "price":
            p["extra"] = {"tests": tests, "verdict_sentence": tech.get("verdict_sentence"), "divergence": tech.get("divergence"),
                          "divergence_sentence": tech.get("divergence_sentence"), "rsi_zone": tech.get("rsi_zone"), "readings": M._readings(tech)}
        elif pid == "watch":
            p["extra"] = {"signposts": signposts, "portfolio": {"current_weight": None, "target_weight": None, "triggers": None}}
        panels.append(p)

    gloss = [g for g in ve.load_glossary() if g["id"] not in _DROP_TERMS] + list(variant.get("glossary_add") or [])
    thesis = bh.thesis(chips, facts, light_map)
    model = {
        "schema_version": M.SCHEMA_VERSION, "meta": meta, "thesis": thesis, "warnings": warnings, "panels": panels, "glossary": gloss,
        "sources": {"provider": "FinEdge API (NSE/BSE company filings and quotes; standalone bank statements)",
                    "snapshot": f"Price {m['price']} on {m.get('price_date')}; report date {meta['report_date']}.",
                    "definitions": [f"{c['label']}: {c['calc']}" for p in panels for c in p["chips"]],
                    "validation": {"errors": [], "warnings": [], "passed": []}},
        "disclaimer": M.DISCLAIMER, "facts": {k: v for k, v in facts.items() if not isinstance(v, dict) and k != "years"},
        "headline_rules": variant.get("headline_rules"),
    }
    model["facts"]["years"] = facts.get("years")
    result = vmod.validate_model(model)
    extra_errors = P.check_cases(cases, facts.get("price"))     # validate.py recomputes EBITDA-based cases only
    result["errors"] += extra_errors
    if not extra_errors and cases:
        result["passed"].append(f"{len(cases)} fair-value cases recompute from their multiples and base values and are measured against the snapshot price")
    model["sources"]["validation"] = result
    for e in result["errors"]:
        model["warnings"].insert(0, {"level": "error", "text": e})
    for w in result["warnings"]:
        model["warnings"].append({"level": "warning", "text": w})
    return model
