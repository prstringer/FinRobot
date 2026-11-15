#!/usr/bin/env python
# coding: utf-8
"""
Cost taxonomy for the report: which labels describe a company's cost structure.

The analysis table keeps stable INTERNAL row names ("Cost of Operations", "SG&A", "Contribution Profit", ...), because many
modules look rows up by name. The taxonomy decides how they are DISPLAYED and described:

* ``resolve(sector, industry, macro_sector)`` -> taxonomy dict (kind, labels, limitation, replacements ...)
* ``rename_rows(df, taxonomy)``                -> copy of a table with display row names
* ``apply_text(obj, taxonomy)``                -> phrase replacements over strings (deep, for dict/list models)
* ``add_split_rows(df, income_df, taxonomy)``  -> adds "Employee Cost" and "Other Operating Expenses" rows (industrial only)

EBITDA is NOT changed by the taxonomy: it is always the Ind-AS figure, revenue from operations less total operating expenses
(see ``EBITDA_DEFINITION``); other income and exceptional items are excluded.
"""

from __future__ import annotations

import copy
import os
import re
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIX = " (INR crore)"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(os.path.join(HERE, "cost_taxonomy.yaml"), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


EBITDA_DEFINITION = load_config()["ebitda_definition"]


def _match(text: str, keywords: list[str]) -> Optional[str]:
    t = (text or "").lower()
    return next((k for k in keywords if k in t), None)


def classify(sector: Optional[str], industry: Optional[str] = None, macro_sector: Optional[str] = None) -> dict:
    """Pick the taxonomy kind from the provider's sector fields. Unknown sectors fall back to generic with a limitation."""
    cfg = load_config()
    sector, industry, macro = sector or "", industry or "", macro_sector or ""
    g = _match(sector, cfg["generic_keywords"])
    if g:
        return {"kind": "generic", "matched": g, "known": True}
    i = _match(sector, cfg["industrial_keywords"]) or _match(industry, cfg["industrial_keywords"]) or _match(macro, cfg["industrial_keywords"])
    if i:
        return {"kind": "industrial", "matched": i, "known": True}
    g2 = _match(industry, cfg["generic_keywords"]) or _match(macro, cfg["generic_keywords"])
    if g2:
        return {"kind": "generic", "matched": g2, "known": True}
    return {"kind": "generic", "matched": None, "known": False}


def resolve(sector: Optional[str] = None, industry: Optional[str] = None, macro_sector: Optional[str] = None,
            kind: Optional[str] = None) -> dict:
    """Taxonomy dict stored in ``final_df.attrs['taxonomy']``, ``ctx['taxonomy']`` and ``model['meta']['taxonomy']``."""
    cfg = load_config()
    c = {"kind": kind, "matched": None, "known": True} if kind else classify(sector, industry, macro_sector)
    k = cfg["kinds"][c["kind"]]
    limitation = k.get("limitation")
    if c["kind"] == "generic" and not c["known"]:
        limitation = k.get("unknown_limitation")
    return {"kind": c["kind"], "label": k["label"], "matched": c["matched"], "sector": sector, "industry": industry,
            "macro_sector": macro_sector, "labels": dict(k["labels"]), "limitation": limitation,
            "ebitda_definition": cfg["ebitda_definition"]}


def taxonomy_from_table(df: Optional[pd.DataFrame]) -> Optional[dict]:
    """Detect the taxonomy from a saved analysis table: the industrial split rows exist only for industrial companies.
    Lets scripts that reload the CSV (no context object) label rows consistently. Returns None for a generic table."""
    if df is None or df.empty or "metrics" not in df.columns:
        return None
    if (df["metrics"] == "Employee Cost").any():
        return {"kind": "industrial"}
    return None


def chart_label(name: str, taxonomy: Optional[dict]) -> str:
    """Short legend label for a row name under the taxonomy (generic: unchanged)."""
    if not is_industrial(taxonomy):
        return name
    return load_config()["kinds"]["industrial"].get("chart_names", {}).get(name, name)


def is_industrial(taxonomy: Optional[dict]) -> bool:
    return bool(taxonomy) and taxonomy.get("kind") == "industrial"


def rename_rows(df: pd.DataFrame, taxonomy: Optional[dict]) -> pd.DataFrame:
    """Copy of ``df`` with display names in the 'metrics' column (or index) for the taxonomy kind. Generic = unchanged."""
    if not is_industrial(taxonomy) or df is None or df.empty:
        return df
    names = load_config()["kinds"]["industrial"]["row_names"]

    def conv(name):
        s = str(name)
        suf = SUFFIX if s.endswith(SUFFIX) else ""
        base = s[: -len(SUFFIX)] if suf else s
        return names.get(base, base) + suf if base in names else s
    out = df.copy()
    if "metrics" in out.columns:
        out["metrics"] = out["metrics"].map(conv)
    else:
        out.index = [conv(i) for i in out.index]
    out.attrs = dict(df.attrs)
    return out


def apply_text(obj: Any, taxonomy: Optional[dict], skip_keys: tuple = ("glossary",)) -> Any:
    """Apply the taxonomy's ordered phrase replacements to every string in ``obj`` (str, list, dict). Identifier-like keys
    (id, key, ...) are left alone. Generic taxonomy returns ``obj`` unchanged."""
    if not is_industrial(taxonomy):
        return obj
    reps = load_config()["kinds"]["industrial"]["replacements"]
    ident = {"id", "key", "yardstick", "verdict", "icon", "kind", "level", "role", "light", "status", "schema_version"}

    def rep(s: str) -> str:
        for a, b in reps:
            if a not in s:
                continue
            lower_b = b[:1].lower() + b[1:] if b[:1].isupper() and not b.split()[0].isupper() else b

            def sub(m, b=b, lower_b=lower_b):
                before = s[: m.start()].rstrip()
                # capitalised at the start of a string or sentence/label, lower-case mid-sentence
                return b if (not before or before[-1] in ".:!?(\"'") else lower_b
            s = re.sub(re.escape(a), sub, s)
        return s

    def walk(o, key=None):
        if isinstance(o, str):
            return o if key in ident else rep(o)
        if isinstance(o, list):
            return [walk(x, key) for x in o]
        if isinstance(o, tuple):
            return tuple(walk(x, key) for x in o)
        if isinstance(o, dict):
            return {k: (v if k in skip_keys else walk(v, k)) for k, v in o.items()}
        return o
    return walk(obj)


def glossary_for(glossary: list[dict], taxonomy: Optional[dict]) -> list[dict]:
    """Glossary with the taxonomy's overrides applied (replacing entries with the same id, adding new ones)."""
    if not is_industrial(taxonomy):
        return glossary
    over = {g["id"]: g for g in load_config()["kinds"]["industrial"]["glossary_overrides"]}
    out = [dict(over.get(g["id"], g)) for g in glossary]
    have = {g["id"] for g in out}
    out += [dict(g) for gid, g in over.items() if gid not in have]
    return out


def notes(taxonomy: Optional[dict]) -> list[str]:
    """Data notes that must travel with the report for this taxonomy."""
    if not taxonomy:
        return []
    out = []
    if taxonomy.get("limitation"):
        out.append(taxonomy["limitation"])
    if is_industrial(taxonomy):
        out.append("Cost labels follow the industrial taxonomy: materials cost, employee cost and other operating expenses. "
                   + taxonomy["ebitda_definition"])
    return out


def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def add_split_rows(final_df: pd.DataFrame, income_df: Optional[pd.DataFrame], taxonomy: Optional[dict]) -> pd.DataFrame:
    """Industrial only: add 'Employee Cost' and 'Other Operating Expenses' rows (absolute rupees) after the SG&A row.

    Actual years come from the income statement. Forecast years keep the last actual employee cost as % of revenue; the other
    operating expenses bucket is the remainder of the forecast SG&A bucket, so it absorbs any margin-cap adjustment and
    employee + other always equals the SG&A row."""
    if not is_industrial(taxonomy) or income_df is None or income_df.empty:
        return final_df
    df = final_df.copy().astype("object")
    df.attrs = dict(final_df.attrs)
    inc = income_df.set_index("year")
    years_a = [c for c in df.columns if isinstance(c, str) and re.fullmatch(r"\d{4}A", c)]
    years_e = [c for c in df.columns if isinstance(c, str) and re.fullmatch(r"\d{4}E", c)]

    def row(metric):
        sel = df[df["metrics"] == metric]
        return None if sel.empty else sel.iloc[0]
    rev, sga = row("Revenue"), row("SG&A")
    if rev is None or sga is None:
        return final_df
    emp, oth = {}, {}
    for c in years_a:
        y = int(c[:4])
        if y in inc.index:
            e = _num(inc.loc[y].get("employeeBenefitExpense"))
            o = _num(inc.loc[y].get("otherExpenses"))
            emp[c], oth[c] = e, o
    last = [c for c in years_a if emp.get(c) is not None and _num(rev[c])]
    ratio = (emp[last[-1]] / _num(rev[last[-1]])) if last else None
    for c in years_e:
        r, s = _num(rev[c]), _num(sga[c])
        if r is None or s is None or ratio is None:
            continue
        e = min(ratio * r, s)
        emp[c], oth[c] = e, s - e
    cols = [c for c in df.columns if c != "metrics"]
    new_rows = []
    for name, vals in (("Employee Cost", emp), ("Other Operating Expenses", oth)):
        rec = {"metrics": name}
        rec.update({c: (round(vals[c], 2) if vals.get(c) is not None else None) for c in cols if c in vals})
        for c in cols:
            rec.setdefault(c, None)
        new_rows.append(rec)
    idx = int(df.index[df["metrics"] == "SG&A"][0]) + 1
    top, bottom = df.iloc[:idx], df.iloc[idx:]
    out = pd.concat([top, pd.DataFrame(new_rows), bottom], ignore_index=True).astype("object")
    out.attrs = dict(final_df.attrs)
    return out
