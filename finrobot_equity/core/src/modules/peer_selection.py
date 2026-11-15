#!/usr/bin/env python
# coding: utf-8
"""
Peer selection: same sub-sector, comparable size.

Why: FinEdge's sector lists are coarse. For MAHASTEEL (market cap about Rs 2,500 Cr, EBITDA margin 2.7%) they returned
tube and pipe makers worth Rs 16,000 to 62,000 Cr with double-digit margins, which makes any relative valuation grid
meaningless. ``select_peers`` keeps a peer only if

  1. it is in the same sub-sector (``subsector_map.yaml``; fallback documented below), and
  2. its market cap is within ``band`` times the subject's market cap (default 0.2x to 5.0x, both ends inclusive).

If fewer than ``min_peers`` survive the result is flagged ``insufficient`` instead of silently widening the net; the
report must then say its relative valuation grid is not reliable. Every rejected candidate is logged with its reason.

Fallback when the subject has no sub-sector tag: the FinEdge ``sub_industry`` list is used as-is (still market-cap banded)
and ``basis`` says so.
"""

from __future__ import annotations

import math
import os
import statistics
from functools import lru_cache
from typing import Optional

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BAND = (0.2, 5.0)
MARGIN_FACTOR = 3.0          # peers' median margin more than 3x (or less than 1/3) of the subject's
MARGIN_GAP_PP = 5.0          # or more than 5 percentage points apart
MAX_POOL = 30                # most candidates we fetch a market cap for (one quote call each)


@lru_cache(maxsize=1)
def load_subsector_map(path: Optional[str] = None) -> dict:
    with open(path or os.path.join(HERE, "subsector_map.yaml"), "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    tag_of: dict[str, str] = {}
    for tag, d in (cfg.get("subsectors") or {}).items():
        for t in d.get("tickers") or []:
            tag_of.setdefault(str(t).upper(), tag)
    return {"subsectors": cfg.get("subsectors") or {}, "tag_of": tag_of, "sector_tags": cfg.get("sector_tags") or {}}


def subsector_of(symbol: str, sector: Optional[str] = None) -> Optional[str]:
    m = load_subsector_map()
    return m["tag_of"].get(symbol.upper()) or (m["sector_tags"].get(sector) if sector else None)


def subsector_label(tag: Optional[str]) -> Optional[str]:
    return (load_subsector_map()["subsectors"].get(tag) or {}).get("label") if tag else None


def _norm(sym: str) -> str:
    sym = str(sym).strip().upper()
    for suffix in (".NS", ".BO"):
        if sym.endswith(suffix):
            sym = sym[: -len(suffix)]
    return sym


def _mcap(client, symbol: str) -> Optional[float]:
    try:
        q = client.quote(symbol) or {}
        v = float(q.get("market_cap") or 0)
        return v if v > 0 else None
    except Exception:
        return None


def _pool(client, symbol: str) -> tuple[list[str], str]:
    """FinEdge candidates: the sub_industry list, else the sector list."""
    for group in ("sub_industry", "sector"):
        try:
            peers = client.peers(symbol, group) or []
        except Exception:
            peers = []
        peers = [_norm(p) for p in peers if _norm(p) != symbol]
        if peers:
            return peers[:MAX_POOL], group
    return [], "none"


def select_peers(symbol: str, client, max_peers: int = 5, band: tuple = DEFAULT_BAND, min_peers: int = 3,
                 explicit: Optional[list[str]] = None) -> dict:
    """Choose peers for ``symbol``. See module docstring. Returns the selection log (JSON-serialisable)."""
    symbol = _norm(symbol)
    lo, hi = band
    subject = _mcap(client, symbol)
    try:
        sector = (client.profile(symbol) or {}).get("sector")
    except Exception:
        sector = None
    tag = subsector_of(symbol, sector)
    out = {"subject": symbol, "subject_market_cap_cr": subject, "band": [lo, hi], "subsector": subsector_label(tag),
           "subsector_tag": tag, "sector": sector, "peers": [], "rejected": [], "insufficient": False, "warnings": [],
           "basis": "", "explicit": bool(explicit)}
    if subject is None:
        out["warnings"].append(f"Market cap of {symbol} is unavailable, so the size band cannot be applied.")

    def ratio(mc):
        return (mc / subject) if (mc and subject) else None

    # user-supplied peers are kept; those outside the band are called out
    if explicit:
        kept = [_norm(p) for p in explicit if _norm(p) != symbol]
        out["peers"] = kept
        out["basis"] = "user-supplied peers (kept as given)"
        for p in kept:
            mc = _mcap(client, p)
            r = ratio(mc)
            ptag = subsector_of(p)
            if r is not None and not (lo <= r <= hi):
                out["warnings"].append(f"{p} has {r:.1f}x the market cap of {symbol}, outside the {lo}x to {hi}x band.")
            if tag and ptag and ptag != tag:
                out["warnings"].append(f"{p} is in a different sub-sector ({subsector_label(ptag)}) from {symbol} ({subsector_label(tag)}).")
        return out

    pool, group = _pool(client, symbol)
    tag_members = [t for t in (load_subsector_map()["subsectors"].get(tag, {}).get("tickers") or []) if t.upper() != symbol] if tag else []
    if tag:
        candidates = list(dict.fromkeys([m.upper() for m in tag_members] + [p for p in pool if subsector_of(p) == tag]))
        out["basis"] = f"sub-sector tag '{subsector_label(tag)}' (subsector_map.yaml), market-cap band {lo}x to {hi}x"
    else:
        candidates = list(pool)
        out["basis"] = (f"FinEdge {group} list, no sub-sector tag for {symbol}; market-cap band {lo}x to {hi}x. "
                        "Add the ticker to subsector_map.yaml for a stricter match")
        out["warnings"].append(f"{symbol} has no sub-sector tag: peers come from the FinEdge {group} list as-is, banded by size only.")

    survivors: list[tuple[float, str, float, float]] = []
    seen = set()
    for p in candidates:
        if p in seen or p == symbol:
            continue
        seen.add(p)
        mc = _mcap(client, p)
        r = ratio(mc)
        if mc is None:
            out["rejected"].append({"ticker": p, "market_cap_cr": None, "ratio": None, "reason": "market cap unavailable"})
        elif r is None:
            survivors.append((0.0, p, mc, float("nan")))
        elif not (lo <= r <= hi):
            out["rejected"].append({"ticker": p, "market_cap_cr": mc, "ratio": round(r, 2),
                                    "reason": f"market cap {r:.2f}x the subject, outside {lo}x to {hi}x"})
        else:
            survivors.append((abs(math.log(r)), p, mc, r))
    # FinEdge-listed names that are not in the subject's sub-sector are logged as rejected (with their size for context)
    if tag:
        for p in pool:
            if p in seen or p == symbol:
                continue
            ptag = subsector_of(p)
            mc = _mcap(client, p)
            r = ratio(mc)
            why = (f"different sub-sector ({subsector_label(ptag)})" if ptag
                   else f"not in sub-sector '{subsector_label(tag)}' (untagged)")
            out["rejected"].append({"ticker": p, "market_cap_cr": mc, "ratio": round(r, 2) if r else None, "reason": why})
    survivors.sort(key=lambda s: (s[0], s[1]))
    chosen = survivors[:max_peers]
    for _, p, mc, r in survivors[max_peers:]:
        out["rejected"].append({"ticker": p, "market_cap_cr": mc, "ratio": None if r != r else round(r, 2),
                                "reason": f"eligible but beyond the {max_peers} closest in size"})
    out["peers"] = [c[1] for c in chosen]
    out["peer_market_caps_cr"] = {c[1]: c[2] for c in chosen}
    if len(out["peers"]) < min_peers:
        out["insufficient"] = True
        out["warnings"].append(
            f"Only {len(out['peers'])} comparable peer(s) found for {symbol} (need {min_peers}) within {lo}x to {hi}x of its market cap"
            + (f" in '{subsector_label(tag)}'" if tag else "")
            + ". The relative valuation grid is not reliable and should not be used for conclusions.")
    return out


def margin_profile_check(subject_margin_pct: Optional[float], peer_margins: dict) -> dict:
    """Compare the subject's EBITDA margin with the peers' median. Mismatch = more than 3x apart or more than 5 points apart."""
    vals = [v for v in (peer_margins or {}).values() if v is not None]
    res = {"subject_margin_pct": subject_margin_pct, "peer_median_margin_pct": None, "ratio": None, "gap_pp": None,
           "mismatch": False, "warning": None, "peers_used": len(vals)}
    if subject_margin_pct is None or not vals:
        return res
    med = statistics.median(vals)
    res["peer_median_margin_pct"] = med
    res["gap_pp"] = abs(med - subject_margin_pct)
    if subject_margin_pct > 0 and med > 0:
        res["ratio"] = max(med / subject_margin_pct, subject_margin_pct / med)
    big_ratio = res["ratio"] is not None and res["ratio"] > MARGIN_FACTOR
    mixed_sign = (subject_margin_pct > 0) != (med > 0)
    if big_ratio or res["gap_pp"] > MARGIN_GAP_PP or mixed_sign:
        res["mismatch"] = True
        res["warning"] = (f"Margin profiles differ: subject EBITDA margin {subject_margin_pct:.1f}% vs peer median {med:.1f}% "
                          f"({res['gap_pp']:.1f} points apart"
                          + (f", {res['ratio']:.1f}x" if res["ratio"] else "")
                          + "). Relative valuation against these peers compares different business models.")
    return res


def summary_lines(sel: dict, margin: Optional[dict] = None) -> list[str]:
    """Readable summary for the console and the log."""
    L = [f"Peer selection for {sel['subject']}: market cap Rs {sel['subject_market_cap_cr'] or 0:,.0f} Cr, band {sel['band'][0]}x to {sel['band'][1]}x",
         f"  Basis: {sel['basis']}"]
    L.append(f"  Kept ({len(sel['peers'])}): " + (", ".join(sel['peers']) if sel['peers'] else "none"))
    for r in sel["rejected"][:12]:
        mc = f"Rs {r['market_cap_cr']:,.0f} Cr" if r.get("market_cap_cr") else "n/a"
        L.append(f"  Rejected {r['ticker']} ({mc}): {r['reason']}")
    if len(sel["rejected"]) > 12:
        L.append(f"  ... and {len(sel['rejected']) - 12} more rejected (see peer_selection.json)")
    L += [f"  WARNING: {w}" for w in sel["warnings"]]
    if margin and margin.get("warning") and margin["warning"] not in sel["warnings"]:
        L.append(f"  WARNING: {margin['warning']}")
    return L


def peer_warnings(sel: dict) -> list[str]:
    """Reader-facing warnings for the report: thin peer set and margin-profile mismatch."""
    out = []
    if sel.get("insufficient"):
        n = len(sel.get("peers") or [])
        out.append(f"Peer set is thin: {n} comparable peer(s) within {sel['band'][0]}x to {sel['band'][1]}x of the company's market cap "
                   f"and sub-sector. Peer-relative valuation (peer medians, premium or discount) is not reliable.")
    mp = sel.get("margin_profile") or {}
    if mp.get("mismatch") and mp.get("warning"):
        out.append(mp["warning"])
    return out


def peer_fact_block(sel: dict) -> str:
    """Facts for the text layer when the peer set cannot support a relative valuation."""
    ws = peer_warnings(sel)
    if not ws:
        return None
    kept = ", ".join(sel.get("peers") or []) or "none"
    rej = len(sel.get("rejected") or [])
    return ("PEER SET LIMITATION (computed): peers kept after the market-cap band and sub-sector filter: " + kept + f" ({rej} candidates rejected). "
            + " ".join(ws) + " Do not draw valuation conclusions from peer multiples or peer medians; say the peer comparison is limited.")
