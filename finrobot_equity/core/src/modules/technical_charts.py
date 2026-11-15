#!/usr/bin/env python
# coding: utf-8
"""Technical chart: price with moving averages, RSI and MACD panels, with divergence marked. Values match technical_analysis."""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .technical_analysis import prepare  # noqa: E402

DEFAULT_SESSIONS = 250


def make_technical_chart(price_df: pd.DataFrame, facts: dict, out_path: str, ticker: str = "",
                         sessions: int = DEFAULT_SESSIONS) -> Optional[str]:
    """Draw the three-panel chart to ``out_path`` and return the path (None when there is too little data)."""
    if not facts or not facts.get("available"):
        return None
    df = prepare(price_df)
    start = max(0, len(df) - sessions)
    d = df.iloc[start:]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1.2, 1.4], "hspace": 0.08})
    ax1.plot(d["date"], d["close"], color="#0f172a", lw=1.6, label="Close")
    for col, colr, lab in (("sma20", "#d4a843", "20-day"), ("sma50", "#2563eb", "50-day"), ("sma200", "#dc2626", "200-day")):
        if d[col].notna().any():
            ax1.plot(d["date"], d[col], color=colr, lw=1.1, label=lab)
    ax1.axhline(facts["week52_high"], color="#94a3b8", ls=":", lw=0.9)
    ax1.text(d["date"].iloc[0], facts["week52_high"], " 52-week high", va="bottom", fontsize=7, color="#64748b")
    ax1.set_title(f"{ticker} technical readings (as of {facts['as_of_date']})", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=7, ncol=4, frameon=False)
    ax1.grid(alpha=0.3)

    ax2.plot(d["date"], d["rsi"], color="#7c3aed", lw=1.2)
    ax2.axhspan(70, 100, color="#fee2e2", alpha=0.5)
    ax2.axhspan(0, 30, color="#dcfce7", alpha=0.5)
    ax2.axhline(60, color="#94a3b8", ls=":", lw=0.8)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI (14)", fontsize=8)
    ax2.grid(alpha=0.3)

    ax3.plot(d["date"], d["macd"], color="#2563eb", lw=1.2, label="MACD")
    ax3.plot(d["date"], d["macd_signal"], color="#d97706", lw=1.0, label="Signal")
    ax3.bar(d["date"], d["macd_hist"], color=["#16a34a" if v >= 0 else "#dc2626" for v in d["macd_hist"]], width=1.0, alpha=0.45)
    ax3.axhline(0, color="#64748b", lw=0.6)
    ax3.set_ylabel("MACD", fontsize=8)
    ax3.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)
    ax3.grid(alpha=0.3)

    # divergence: join the two swing points in price and in each indicator panel (indices are into the full frame)
    idx = facts.get("_divergence_idx", {})
    for kind, colr in (("bearish", "#dc2626"), ("bullish", "#16a34a")):
        for (i1, i2, j1, j2), item in zip(idx.get(kind, []), facts["divergence"].get(kind, [])):
            ax1.plot([df["date"].iloc[i1], df["date"].iloc[i2]], [df["close"].iloc[i1], df["close"].iloc[i2]],
                     color=colr, lw=1.6, ls="--", marker="o", ms=4)
            panel, col = (ax3, "macd") if item["indicator"] == "MACD" else (ax2, "rsi")
            panel.plot([df["date"].iloc[j1], df["date"].iloc[j2]], [df[col].iloc[j1], df[col].iloc[j2]],
                       color=colr, lw=1.6, ls="--", marker="o", ms=4)
            ax1.annotate(f"{kind} divergence ({item['indicator']})", (df["date"].iloc[i2], df["close"].iloc[i2]),
                         textcoords="offset points", xytext=(-95, 12 if kind == "bearish" else -22), fontsize=7, color=colr)
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
