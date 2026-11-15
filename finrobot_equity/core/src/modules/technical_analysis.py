#!/usr/bin/env python
# coding: utf-8
"""
Technical readings, four independent tests, divergence detection and a price-vs-earnings strip.

Pure functions: a price DataFrame (date, open, high, low, close, volume) in, plain dicts out. No ticker or currency
assumptions. Everything the report prints about technicals is computed here, so text and charts agree.

Definitions used (also returned in the facts so the report can state them):
* SMA20/50/200: simple means of the close; RSI(14) uses Wilder smoothing; MACD is EMA12 - EMA26 with a 9-period
  EMA signal line (non-adjusted EMAs).
* 52-week window: the last 250 sessions (highs and lows, not closes).
* Crossover: the last change of sign of SMA50 - SMA200 (golden = 50 above 200, death = 50 below 200).
* Current trend window: from the last session on which the close crossed above the SMA50; if that is more than 120
  sessions ago, absent, or shorter than 20 sessions, the last 60 sessions are used. The prior base is the same
  number of sessions immediately before it.
* Swing point: a bar whose close is the highest (lowest) within +/- ``SWING_WINDOW`` sessions. A swing therefore
  needs ``SWING_WINDOW`` later sessions to be confirmed.
* Divergence: of the last two swing highs (lows) in price within the last ``DIVERGENCE_LOOKBACK`` sessions and at
  least ``MIN_SWING_GAP`` sessions apart, price makes a higher high (lower low) while the indicator, taken as its
  extreme within +/- 3 sessions of each swing, makes a lower high (higher low). Checked for MACD and RSI.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

SWING_WINDOW = 5
DIVERGENCE_LOOKBACK = 160
MIN_SWING_GAP = 10
MIN_ROWS = 30
YEAR_SESSIONS = 250


# --------------------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------------------
def _f(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _r(x, d=2) -> Optional[float]:
    v = _f(x)
    return None if v is None else round(v, d)


def prepare(price_df: pd.DataFrame) -> pd.DataFrame:
    """Sorted copy with parsed dates, numeric columns, and the indicator columns added."""
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    for col in ("high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    c = df["close"]
    for n in (20, 50, 200):
        df[f"sma{n}"] = c.rolling(n).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).where(loss != 0, 100.0).where(gain.notna())
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def rsi_zone(rsi: Optional[float]) -> str:
    """C4 zones: never a plain 'bullish' between 60 and 70."""
    if rsi is None:
        return "n/a"
    if rsi < 30:
        return "oversold"
    if rsi < 60:
        return "neutral"
    if rsi <= 70:
        return "strong, approaching overbought"
    return "overbought"


# --------------------------------------------------------------------------------------
# swings and divergence
# --------------------------------------------------------------------------------------
def swing_points(series: pd.Series, window: int = SWING_WINDOW, kind: str = "high") -> list[int]:
    """Indices where the value is the strict extreme of its +/- window neighbourhood (confirmed swings only)."""
    vals = series.to_numpy(dtype=float)
    out = []
    for i in range(window, len(vals) - window):
        seg = vals[i - window:i + window + 1]
        if np.isnan(seg).any():
            continue
        if kind == "high" and vals[i] == seg.max() and (seg == vals[i]).sum() == 1:
            out.append(i)
        if kind == "low" and vals[i] == seg.min() and (seg == vals[i]).sum() == 1:
            out.append(i)
    return out


def _indicator_extreme(ind: pd.Series, i: int, kind: str, span: int = 3) -> tuple[int, float]:
    lo, hi = max(0, i - span), min(len(ind), i + span + 1)
    seg = ind.iloc[lo:hi]
    j = int(seg.idxmax() if kind == "high" else seg.idxmin())
    return j, float(ind.loc[j])


def find_divergence(df: pd.DataFrame) -> dict:
    """Bearish (highs) and bullish (lows) divergence of price against MACD and RSI. See module docstring for the rule."""
    result = {"bearish": [], "bullish": [], "rule": (
        f"last two swing points in price (+/-{SWING_WINDOW} sessions, at least {MIN_SWING_GAP} apart, within the last "
        f"{DIVERGENCE_LOOKBACK} sessions); indicator extreme within +/-3 sessions of each")}
    start = max(0, len(df) - DIVERGENCE_LOOKBACK)
    for kind, label in (("high", "bearish"), ("low", "bullish")):
        swings = [i for i in swing_points(df["close"], kind=kind) if i >= start]
        # last two swings far enough apart
        pair = None
        for a in range(len(swings) - 1, 0, -1):
            b = a - 1
            if swings[a] - swings[b] >= MIN_SWING_GAP:
                pair = (swings[b], swings[a])
                break
        if not pair:
            continue
        i1, i2 = pair
        p1, p2 = float(df["close"].iloc[i1]), float(df["close"].iloc[i2])
        price_ok = p2 > p1 if kind == "high" else p2 < p1
        if not price_ok:
            continue
        for name, col in (("MACD", "macd"), ("RSI", "rsi")):
            j1, v1 = _indicator_extreme(df[col], i1, kind)
            j2, v2 = _indicator_extreme(df[col], i2, kind)
            ind_ok = v2 < v1 if kind == "high" else v2 > v1
            if ind_ok:
                result[label].append({
                    "indicator": name, "kind": label,
                    "price_date_1": df["date"].iloc[i1].strftime("%Y-%m-%d"), "price_1": _r(p1),
                    "price_date_2": df["date"].iloc[i2].strftime("%Y-%m-%d"), "price_2": _r(p2),
                    "indicator_date_1": df["date"].iloc[j1].strftime("%Y-%m-%d"), "indicator_1": _r(v1),
                    "indicator_date_2": df["date"].iloc[j2].strftime("%Y-%m-%d"), "indicator_2": _r(v2),
                    "idx": (i1, i2, j1, j2),
                })
    return result


def divergence_sentence(div: dict) -> str:
    """One plain sentence, or a statement that none was found."""
    for kind in ("bearish", "bullish"):
        items = div.get(kind) or []
        if items:
            names = " and ".join(d["indicator"] for d in items)
            d = items[0]
            what = ("price made a higher high" if kind == "bearish" else "price made a lower low")
            ind = ("a lower high" if kind == "bearish" else "a higher low")
            vals = "; ".join(f"{x['indicator']} {x['indicator_1']} to {x['indicator_2']}" for x in items)
            return (f"{kind.capitalize()} divergence: {what} ({d['price_1']} on {d['price_date_1']} to {d['price_2']} on "
                    f"{d['price_date_2']}) while {names} made {ind} ({vals}).")
    return "No divergence between price and MACD or RSI at the last two swing points."


# --------------------------------------------------------------------------------------
# facts
# --------------------------------------------------------------------------------------
def _last_crossover(df: pd.DataFrame) -> Optional[dict]:
    if df["sma200"].notna().sum() < 2:
        return None
    diff = (df["sma50"] - df["sma200"]).dropna()
    sign = np.sign(diff)
    sign = sign[sign != 0]
    chg = sign[sign.diff().fillna(0) != 0]
    if chg.empty:
        return None
    i = chg.index[-1]
    return {"date": df["date"].loc[i].strftime("%Y-%m-%d"), "direction": "golden" if sign.loc[i] > 0 else "death",
            "sessions_ago": int(len(df) - 1 - i)}


def _trend_window(df: pd.DataFrame) -> tuple[int, str]:
    """Start index of the current trend window and a plain description of the definition used."""
    above = (df["close"] > df["sma50"]) & (df["close"].shift(1) <= df["sma50"].shift(1))
    idx = above[above].index
    n = len(df)
    if len(idx) and 20 <= n - idx[-1] <= 120:
        return int(idx[-1]), f"since the last close above the 50-day average ({df['date'].loc[idx[-1]].strftime('%Y-%m-%d')})"
    return max(0, n - 60), "the last 60 sessions"


def compute_technical_facts(price_df: pd.DataFrame, eps_info: Optional[dict] = None) -> dict:
    """Named readings, tests, divergence and (optionally) the price-vs-earnings strip. ``available`` is False if too short."""
    if price_df is None or len(price_df) < MIN_ROWS:
        return {"available": False, "reason": f"fewer than {MIN_ROWS} price sessions"}
    df = prepare(price_df)
    if len(df) < MIN_ROWS:
        return {"available": False, "reason": f"fewer than {MIN_ROWS} price sessions"}
    last, n = df.iloc[-1], len(df)
    price = float(last["close"])
    win = df.tail(YEAR_SESSIONS)
    hi, lo = float(win["high"].max()), float(win["low"].min())
    f: dict = {
        "available": True, "as_of_date": last["date"].strftime("%Y-%m-%d"), "sessions": n, "price": _r(price),
        "week52_high": _r(hi), "week52_low": _r(lo),
        "week52_position_pct": _r((price - lo) / (hi - lo) * 100, 1) if hi > lo else None,
        "pct_below_52w_high": _r((price / hi - 1) * 100, 1),
        "window_note": f"52-week range uses the last {min(n, YEAR_SESSIONS)} sessions",
    }
    for k in (20, 50, 200):
        v = _f(last[f"sma{k}"])
        f[f"sma{k}"] = _r(v)
        f[f"price_vs_sma{k}_pct"] = _r((price / v - 1) * 100, 1) if v else None
    f["crossover_50_200"] = _last_crossover(df)

    r90 = df.tail(90)
    rsi_now = _f(last["rsi"])
    f.update({"rsi14": _r(rsi_now, 1), "rsi_zone": rsi_zone(rsi_now), "rsi_peak_90d": _r(r90["rsi"].max(), 1),
              "rsi_days_above_70_90d": int((r90["rsi"] > 70).sum())})
    f.update({"macd": _r(last["macd"]), "macd_signal": _r(last["macd_signal"]), "macd_histogram": _r(last["macd_hist"]),
              "macd_peak_90d": _r(r90["macd"].max()),
              "macd_peak_90d_date": r90["date"].loc[r90["macd"].idxmax()].strftime("%Y-%m-%d")})

    t0, tdef = _trend_window(df)
    length = n - t0
    # medians, not means: one-off block-trade or index-rebalancing days would otherwise dominate the comparison
    trend_vol = float(df["volume"].iloc[t0:].median())
    base = df["volume"].iloc[max(0, t0 - length):t0]
    base_vol = float(base.median()) if len(base) else None
    avg20 = float(df["volume"].tail(20).mean())
    f.update({"volume_latest": _r(last["volume"], 0), "volume_avg_20d": _r(avg20, 0),
              "volume_latest_vs_20d": _r(last["volume"] / avg20, 2) if avg20 else None,
              "volume_trend_avg": _r(trend_vol, 0), "volume_base_avg": _r(base_vol, 0) if base_vol else None,
              "volume_trend_vs_base": _r(trend_vol / base_vol, 2) if base_vol else None,
              "volume_trend_window": f"median daily volume over {length} sessions, {tdef}; base = median over the {len(base)} sessions before it"})

    div = find_divergence(df)
    f["divergence"] = {k: ([{kk: vv for kk, vv in d.items() if kk != "idx"} for d in v] if isinstance(v, list) else v)
                       for k, v in div.items()}
    f["divergence_sentence"] = divergence_sentence(div)
    f["_divergence_idx"] = {k: [d["idx"] for d in div[k]] for k in ("bearish", "bullish")}

    f["tests"] = four_tests(f)
    f["verdict_sentence"] = verdict_sentence(f)
    if eps_info:
        f["price_vs_earnings"] = price_vs_earnings(df, **eps_info)
    return f


# --------------------------------------------------------------------------------------
# four independent tests
# --------------------------------------------------------------------------------------
def four_tests(f: dict) -> list[dict]:
    """Trend, Momentum, Participation, Stretch. Each is judged on its own evidence; nothing is counted or voted."""
    p, s50, s200 = f["price"], f.get("sma50"), f.get("sma200")
    cross = f.get("crossover_50_200")
    # Trend
    if s50 and s200:
        if p > s50 > s200:
            t_state = "Up"
        elif p < s50 < s200:
            t_state = "Down"
        else:
            t_state = "Mixed"
        ev = f"price {p} vs 50-day {s50} and 200-day {s200}"
        if cross:
            ev += f"; last {cross['direction']} cross {cross['date']}"
    elif s50:
        t_state = "Up" if p > s50 else "Down"
        ev = f"price {p} vs 50-day {s50}; fewer than 200 sessions of history"
    else:
        t_state, ev = "n/a", "not enough history"
    # Momentum
    rsi, macd_up = f.get("rsi14"), (f.get("macd") is not None and f["macd"] > f["macd_signal"])
    if rsi is None:
        m_state = "n/a"
    elif rsi > 55 and macd_up:
        m_state = "Up, near overbought" if rsi >= 60 else "Up"
    elif rsi < 45 and not macd_up:
        m_state = "Down, near oversold" if rsi <= 40 else "Down"
    else:
        m_state = "Mixed"
    if m_state != "n/a" and (f["divergence"].get("bearish")) and m_state.startswith("Up"):
        m_state += ", fading (bearish divergence)"
    m_ev = f"RSI {rsi} ({f['rsi_zone']}); MACD {f['macd']} vs signal {f['macd_signal']}"
    # Participation
    r = f.get("volume_trend_vs_base")
    if r is None:
        pa_state, pa_ev = "n/a", "no prior base to compare"
    else:
        pa_state = "Strong" if r >= 1.15 else ("Weak" if r <= 0.85 else "Normal")
        pa_ev = f"median volume in the trend is {r}x the prior base; latest session {f['volume_latest_vs_20d']}x the 20-day average"
    # Stretch
    d200 = f.get("price_vs_sma200_pct")
    if d200 is None:
        st_state, st_ev = "n/a", "no 200-day average yet"
    else:
        st_state = ("Very high" if d200 >= 40 else "High" if d200 >= 25 else "Moderate" if d200 >= 10
                    else "Normal" if d200 > -10 else "Depressed")
        below = abs(f.get('pct_below_52w_high') or 0)
        st_ev = (f"price is {abs(d200)}% {'above' if d200 >= 0 else 'below'} the 200-day average and {below}% below the 52-week high")
    return [{"test": "Trend", "measures": "price vs 50/200-day averages, crossover", "state": t_state, "evidence": ev},
            {"test": "Momentum", "measures": "RSI and MACD together", "state": m_state, "evidence": m_ev},
            {"test": "Participation", "measures": "volume vs average, trend vs base", "state": pa_state, "evidence": pa_ev},
            {"test": "Stretch", "measures": "% above the 200-day average", "state": st_state, "evidence": st_ev}]


def verdict_sentence(f: dict) -> str:
    """One plain sentence built from the four states; no em-dashes, no vote counting."""
    t = {x["test"]: x["state"] for x in f["tests"]}
    trend = t["Trend"]
    head = {"Up": "Uptrend intact", "Down": "Downtrend in place", "Mixed": "Trend unclear", "n/a": "Trend cannot be judged"}[trend]
    quals = []
    if trend == "Up" and t["Stretch"] in ("High", "Very high"):
        quals.append("late")
    if t["Stretch"] in ("High", "Very high"):
        quals.append("stretched")
    elif t["Stretch"] == "Depressed":
        quals.append("deeply discounted to its long-term average")
    if t["Participation"] == "Weak":
        quals.append("thinly traded")
    tail = ""
    if f["divergence"].get("bearish") and trend == "Up":
        tail = "momentum fading against price"
    elif f["divergence"].get("bullish") and trend == "Down":
        tail = "selling momentum easing"
    elif t["Momentum"].startswith("Up, near overbought"):
        quals.append("close to overbought")
    text = head
    if quals:
        text += (", but " if trend == "Up" else ": ") + (", ".join(quals[:-1]) + (" and " if len(quals) > 1 else "") + quals[-1])
    if tail:
        text += ", with " + tail if quals else ": " + tail
    return text + "."


# --------------------------------------------------------------------------------------
# price vs earnings
# --------------------------------------------------------------------------------------
def _pct_change(new, old) -> Optional[float]:
    new, old = _f(new), _f(old)
    if new is None or old is None or old <= 0:
        return None
    return (new / old - 1) * 100


def price_vs_earnings(price_df: pd.DataFrame, eps_fy_now: float, eps_fy_prev: float, eps_ttm: Optional[float] = None,
                      eps_ttm_year_ago: Optional[float] = None) -> dict:
    """1-year price change next to EPS changes, with one sentence on whether price is ahead of or behind earnings.

    EPS changes are shown for the last full year (vs the year before) and for TTM (vs the same TTM a year ago when
    given, else vs the last full year). A change is n/a when the base EPS is zero or negative.
    """
    df = price_df if "sma20" in price_df.columns else prepare(price_df)
    last = df.iloc[-1]
    target = last["date"] - pd.Timedelta(days=365)
    prior = df[df["date"] <= target]
    px_chg = _pct_change(last["close"], prior["close"].iloc[-1]) if len(prior) else None
    fy_chg = _pct_change(eps_fy_now, eps_fy_prev)
    ttm_base, ttm_base_label = (eps_ttm_year_ago, "TTM a year ago") if eps_ttm_year_ago is not None else (eps_fy_now, "last full year")
    ttm_chg = _pct_change(eps_ttm, ttm_base) if eps_ttm is not None else None
    ref = ttm_chg if ttm_chg is not None else fy_chg
    ref_label = "TTM EPS" if ttm_chg is not None else "last-year EPS"
    if px_chg is None:
        sentence = "Less than a year of price history, so the price move cannot be set against earnings."
        verdict = "n/a"
    elif ref is None:
        sentence = f"The share price moved {px_chg:+.0f}% in a year, but EPS has no positive base to compare against."
        verdict = "n/a"
    else:
        gap = px_chg - ref
        verdict = "ahead" if gap > 15 else ("behind" if gap < -15 else "in line")
        word = {"ahead": "ahead of", "behind": "behind", "in line": "broadly in line with"}[verdict]
        base_note = f" (against the {ttm_base_label})" if ref_label == "TTM EPS" else ""
        sentence = (f"The share price moved {px_chg:+.0f}% in a year while {ref_label}{base_note} moved {ref:+.0f}%, so the price is "
                    f"{word} earnings ({gap:+.0f} points).")
    return {"price_change_1y_pct": _r(px_chg, 1), "eps_fy_now": _r(eps_fy_now), "eps_fy_prev": _r(eps_fy_prev),
            "eps_fy_change_pct": _r(fy_chg, 1), "eps_ttm": _r(eps_ttm), "eps_ttm_base": _r(ttm_base),
            "eps_ttm_base_label": ttm_base_label, "eps_ttm_change_pct": _r(ttm_chg, 1),
            "price_vs_earnings": verdict, "sentence": sentence}


# --------------------------------------------------------------------------------------
# render helpers (data only)
# --------------------------------------------------------------------------------------
def readings_table_rows(f: dict) -> list[tuple[str, str, str]]:
    """(label, value, note) rows for the readings table printed next to the chart."""
    if not f.get("available"):
        return [("Technical readings", "n/a", f.get("reason", "unavailable"))]
    x = f.get("crossover_50_200")
    cross = (f"{x['direction'].capitalize()} cross on {x['date']}", f"{x['sessions_ago']} sessions ago") if x else ("none in the history", "needs 200+ sessions")
    pct = lambda v: "n/a" if v is None else f"{v:+.1f}%"
    return [
        ("Price", f"{f['price']}", f"close on {f['as_of_date']}"),
        ("52-week high / low", f"{f['week52_high']} / {f['week52_low']}", f.get("window_note", "")),
        ("Position in 52-week range", f"{f['week52_position_pct']}%", f"{f['pct_below_52w_high']}% from the high"),
        ("20-day average", f"{f['sma20']}", f"price {pct(f['price_vs_sma20_pct'])}"),
        ("50-day average", f"{f['sma50']}", f"price {pct(f['price_vs_sma50_pct'])}"),
        ("200-day average", f"{f['sma200']}", f"price {pct(f['price_vs_sma200_pct'])}"),
        ("Last 50/200 crossover", cross[0], cross[1]),
        ("RSI (14)", f"{f['rsi14']}", f"{f['rsi_zone']}; 90-day peak {f['rsi_peak_90d']}; {f['rsi_days_above_70_90d']} days above 70"),
        ("MACD / signal / histogram", f"{f['macd']} / {f['macd_signal']} / {f['macd_histogram']}",
         f"90-day MACD peak {f['macd_peak_90d']} on {f['macd_peak_90d_date']}"),
        ("Volume latest / 20-day average", f"{f['volume_latest']:,.0f} / {f['volume_avg_20d']:,.0f}",
         f"latest is {f['volume_latest_vs_20d']}x the average"),
        ("Volume: trend vs prior base",
         "n/a" if f["volume_trend_vs_base"] is None else f"{f['volume_trend_avg']:,.0f} vs {f['volume_base_avg']:,.0f}",
         f"{f['volume_trend_vs_base']}x; {f['volume_trend_window']}" if f["volume_trend_vs_base"] is not None else f["volume_trend_window"]),
        ("Divergence", "yes" if (f["divergence"]["bearish"] or f["divergence"]["bullish"]) else "none", f["divergence_sentence"]),
    ]


def tests_rows(f: dict) -> list[tuple[str, str, str, str]]:
    """(test, measures, state, evidence) rows, then the verdict sentence is in ``f['verdict_sentence']``."""
    if not f.get("available"):
        return []
    return [(t["test"], t["measures"], t["state"], t["evidence"]) for t in f["tests"]]
