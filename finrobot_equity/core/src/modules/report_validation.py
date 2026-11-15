#!/usr/bin/env python
# coding: utf-8
"""
Pre-build validation of one report: data reconciliation, text-versus-source agreement and output hygiene.

``validate_report`` returns ``{"errors": [...], "warnings": [...], "passed": [...]}``. Errors mean a figure in the report
cannot be trusted (a reconciliation failed, a 10x unit slip, a stray HTML entity, a peer or CAGR statement that
disagrees with the data). Warnings are softer (an unlisted figure, an em-dash, a fact repeated too often).
Callers either stop the build (strict mode) or print a visible warning banner on the report.
"""

from __future__ import annotations

import re
from typing import Optional

from . import report_context as rc

_ENTITY_BAD = re.compile(r"&amp;amp;|&(?!(?:amp|lt|gt|quot|apos|nbsp|middot|#\d+|#x[0-9A-Fa-f]+);)[A-Za-z][A-Za-z0-9]*;")
_MULT = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s?x\b")
_CRORE = re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:crore|Cr\b)")


def allowed_multiples(ctx: dict) -> set[float]:
    """Every multiple the report is allowed to quote (all labelled variants, peers, medians, forecast and grid)."""
    out: set[float] = set()

    def add(v):
        if v is not None:
            out.add(round(float(v), 1))
    cur = ctx["current_valuation"]
    for k in ("ev_ebitda_latest_fy", "ev_ebitda_ttm", "pe_latest_fy_at_current_price", "pe_ttm_computed", "pe_ttm_api", "pb"):
        add(cur.get(k))
    for r in ctx["fy_series"].values():
        add(r.get("pe"))
        add(r.get("ev_ebitda"))
    for e in ctx["model_forecast"].values():
        add(e.get("pe_at_current_price"))
    pe = ctx["peers"].get("ev_ebitda")
    if pe:
        for v in pe["values"].values():
            add(v)
        add(pe["median_excl_subject"])
        add(pe["median_incl_subject"])
    pc = ctx["peers"].get("current")
    if pc:
        for r in pc["rows"].values():
            add(r.get("ev_ebitda_fy"))
            add(r.get("ev_ebitda_ttm"))
        for k in ("median_excl_subject_fy", "median_incl_subject_fy", "median_excl_subject_ttm", "median_incl_subject_ttm"):
            add(pc.get(k))
    for g in ctx["fair_value"]["grid"]:
        out.add(float(g["multiple"]))
    for v in (ctx["derived"].get("interest_coverage_ebit_x") or {}).values():
        add(v)
    for r in ((ctx.get("insights") or {}).get("peer_statements") or {}).get("rank", {}).values():
        add(r.get("diff_x"))
    for v in (((ctx.get("extras") or {}).get("technical") or {}).get("tests") or []):
        pass
    ind = (ctx.get("insights") or {}).get("verdicts") or {}
    for v in ind.values():
        s = v["scale"]
        for k in ("value", "range_min", "range_max", "peer_median"):
            add(s.get(k))
    return out


def _sections_with(texts: dict[str, str], needle: str) -> list[str]:
    return [k for k, t in texts.items() if needle in (t or "")]


def validate_report(ctx: Optional[dict], texts: dict[str, str], html: Optional[str] = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    passed: list[str] = []
    all_text = "\n".join(t or "" for t in texts.values())

    if ctx:
        failed = rc.failed_checks(ctx)
        for c in failed:
            errors.append(f"Reconciliation failed: {c['name']} (expected {c['expected']}, got {c['actual']})")
        if not failed:
            passed.append(f"{sum(1 for c in ctx['checks'] if c['ok'])} reconciliation checks passed "
                          f"(price x shares = market cap, EV = market cap + net debt, EPS x shares = PAT, P/E = price / EPS, forecast bridge, peer medians)")

        # monetary values in text against the source tables
        slips = 0
        for sec, t in texts.items():
            for w in rc.lint_narrative_units(t, ctx):
                errors.append(f"{sec}: {w}")
                slips += 1
            canon = list(rc._canonical_crore_values(ctx).values())
            for m in _CRORE.finditer(t or ""):
                n = rc._f(m.group(1).replace(",", ""))
                if n and canon and not any(abs(n - v) <= max(0.01 * v, 0.6) for v in canon):
                    # already reported as a unit slip when it is exactly 10x; otherwise a softer warning
                    if not any(rc.is_unit_slip(n, v) for v in canon):
                        warnings.append(f"{sec}: amount '{m.group(0)}' matches no source figure within 1%")
        if not slips:
            passed.append("No factor-of-10 unit slips in any section")

        # multiples must be labelled variants that exist in the data
        allowed = allowed_multiples(ctx)
        loose = 0
        for sec, t in texts.items():
            for m in _MULT.finditer(t or ""):
                v = round(float(m.group(1)), 1)
                if not any(abs(v - a) <= 0.05 for a in allowed) and v >= 3:
                    warnings.append(f"{sec}: multiple '{m.group(0)}' is not one of the computed figures")
                    loose += 1
        if not loose:
            passed.append("Every multiple quoted in the text is a computed, labelled figure")

        # peer medians in text
        meds = [x for x in (ctx["peers"].get("ev_ebitda") or {}).items() if x[0].startswith("median") and isinstance(x[1], (int, float))]
        med_vals = {round(v, 1) for _, v in meds if v is not None}
        pc = ctx["peers"].get("current") or {}
        med_vals |= {round(pc[k], 1) for k in pc if k.startswith("median") and isinstance(pc[k], (int, float))}
        grid_mults = {float(g["multiple"]) for g in ctx["fair_value"]["grid"]}
        allowed_for_median = {a for a in allowed if a not in grid_mults}   # fair-value grid multiples are assumptions, not medians
        bad_med = 0
        for sec, t in texts.items():
            for sent in re.split(r"(?<=[.!?])\s+", t or ""):
                if "median" in sent.lower() and "ebitda" in sent.lower():   # EV/EBITDA medians only (other ratios have their own peer medians)
                    for m in _MULT.finditer(sent):
                        v = round(float(m.group(1)), 1)
                        peers_vals = {round(x, 1) for x in (ctx["peers"].get("ev_ebitda") or {}).get("values", {}).values() if x is not None}
                        # the sentence may also quote the subject's own multiple; only flag numbers that are neither a median nor a table value
                        if v not in med_vals and v not in allowed_for_median:
                            errors.append(f"{sec}: median sentence quotes {m.group(0)}, not a computed median ({sorted(med_vals)})")
                            bad_med += 1
        if not bad_med:
            passed.append("Every peer-median sentence agrees with the peer tables")

        # revenue CAGR and growth statements
        cagr = None
        try:
            for k, v in (ctx.get("insights") or {}).get("volatility", {}).items():
                pass
        except Exception:
            pass
        cagr = ctx.get("derived", {}).get("revenue_cagr_pct")
        growth_vals = ctx.get("derived", {}).get("revenue_growth_pct") or []
        bad_g = 0
        for sec, t in texts.items():
            for m in re.finditer(r"(\d+(?:\.\d+)?)%\s+(?:revenue\s+)?CAGR|CAGR of (?:about |roughly )?(\d+(?:\.\d+)?)%", t or ""):
                val = float(m.group(1) or m.group(2))
                if cagr is not None and abs(val - abs(cagr)) > 0.15:
                    errors.append(f"{sec}: CAGR {val}% does not recompute from the table ({cagr:.1f}%)")
                    bad_g += 1
            for m in re.finditer(r"revenue\s+(?:fell|declined|dropped|grew|rose|increased)[^.]{0,40}?(\d+(?:\.\d+)?)%", t or "", re.I):
                val = float(m.group(1))
                if growth_vals and not any(abs(val - abs(g)) <= 0.15 for g in growth_vals):
                    errors.append(f"{sec}: revenue change of {val}% matches no year in the table")
                    bad_g += 1
        if not bad_g:
            passed.append("Every stated revenue growth rate and CAGR recomputes from the table")

        # per-share figures (EPS, prices) must be copied exactly, not rounded differently
        ps_allowed = set()
        cur_, mk = ctx["current_valuation"], ctx["market"]
        for v in (cur_.get("eps_latest_fy"), cur_.get("eps_ttm"), mk.get("price"), mk.get("week52_low"), mk.get("week52_high")):
            if v is not None:
                ps_allowed.add(round(float(v), 2))
        for r in ctx["fy_series"].values():
            for k in ("eps", "avg_price"):
                if r.get(k) is not None:
                    ps_allowed.add(round(float(r[k]), 2))
        for e in ctx["model_forecast"].values():
            if e.get("eps") is not None:
                ps_allowed.add(round(float(e["eps"]), 2))
        for g in ctx["fair_value"]["grid"]:
            ps_allowed.add(round(float(g["implied_price"]), 2))
        for k, v in (ctx.get("technicals") or {}).items():
            if isinstance(v, (int, float)) and "pct" not in k:
                ps_allowed.add(round(float(v), 2))
        g_ = ctx.get("sentiment_gate") or {}                       # the gate quotes its implied and threshold prices
        for k_ in ("implied_price", "threshold_price", "current_price"):
            if g_.get(k_) is not None:
                ps_allowed.add(round(float(g_[k_]), 2))
        ol = (ctx.get("insights") or {}).get("operating_leverage") or {}
        if ol.get("ebitda_per_100_revenue") is not None:
            ps_allowed |= {round(abs(float(ol["ebitda_per_100_revenue"])), 2), 100.0}
        bad_ps = 0
        for sec, t in texts.items():
            for m in re.finditer(r"(?:Rs\.?|INR)\s?(\d[\d,]*(?:\.\d+)?)(?![\d,.])(?!\s*(?:crore|Cr\b|lakh|bn|mn))", t or ""):
                raw = m.group(1).replace(",", "")
                v = float(raw)
                if v >= 1000:
                    continue
                if not any(abs(v - a) <= 0.006 or (abs(v - a) <= 0.5 and round(a) == round(v) and "." not in raw) for a in ps_allowed):
                    warnings.append(f"{sec}: per-share figure 'Rs {raw}' does not match a computed value exactly")
                    bad_ps += 1
        if not bad_ps:
            passed.append("Every per-share figure matches a computed value")

        # repetition of key facts
        key_numbers = []
        ins = ctx.get("insights") or {}
        for k in ("revenue_growth", "sga_pct"):
            v = (ins.get("verdicts") or {}).get(k)
            if v:
                key_numbers.append(f"{v['scale']['value']:.1f}%")
        rep = 0
        for num in key_numbers:
            secs = _sections_with(texts, num)
            if len(secs) > 2:
                warnings.append(f"Key figure {num} is stated in {len(secs)} sections ({', '.join(secs)}); it should appear in full once")
                rep += 1
        if not rep:
            passed.append("No key figure is repeated across more than two sections")

    # output hygiene
    ent = _ENTITY_BAD.findall(all_text + ("\n" + html if html else ""))
    if ent:
        errors.append(f"Raw or broken HTML entities in the output: {sorted(set(ent))[:6]}")
    else:
        passed.append("No raw HTML entities in the output")
    if re.search(r"[—–]", all_text):
        warnings.append("Em-dash or en-dash present in generated text")
    return {"errors": errors, "warnings": warnings, "passed": passed}


def banner_text(result: dict) -> Optional[str]:
    """Visible banner text: only for errors (a figure that cannot be trusted). Soft warnings stay in validation_report.md."""
    if not result["errors"]:
        return None
    n_e, n_w = len(result["errors"]), len(result["warnings"])
    return f"VALIDATION WARNING: {n_e} error(s) and {n_w} warning(s) found before this report was built. First: {result['errors'][0]}"


def banner_html(result: dict) -> str:
    b = banner_text(result)
    if not b:
        return ""
    items = "".join(f"<li>{x}</li>" for x in result["errors"][:8])
    return ('<div style="background:#fef2f2;border:2px solid #dc2626;color:#991b1b;padding:10px 14px;margin:8px 12px;'
            f'font-size:12px;font-weight:600;">{b}<ul style="font-weight:400;margin:6px 0 0 18px;">{items}</ul></div>')


def write_validation_report(result: dict, path: str) -> None:
    L = ["# Validation report", "", f"Errors: {len(result['errors'])}. Warnings: {len(result['warnings'])}.", ""]
    for title, key in (("Errors", "errors"), ("Warnings", "warnings"), ("Passed", "passed")):
        L += [f"## {title}", ""] + ([f"* {x}" for x in result[key]] or ["* none"]) + [""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
