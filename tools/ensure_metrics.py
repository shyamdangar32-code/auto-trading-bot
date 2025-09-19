#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build/refresh metrics.json for a reports directory.

Reads trades.csv (and optionally equity curve) to compute:
- trades, win_rate, roi_pct, final_capital
- max_dd_pct, time_dd_bars
- profit_factor, rr (avg win / avg loss)
- sharpe_ratio  (per-trade; sqrt(N) scaled)

Design notes:
- Auto-detect PnL & Status columns
- Handles Inf/NaN robustly and rounds nicely
- Safe when no trades / empty file
"""

import os, sys, json, math
import pandas as pd
import numpy as np

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))

def eprint(*a): print(*a, file=sys.stderr)

def _pick_col(df: pd.DataFrame, keys):
    """Return first matching column (case-insensitive) from keys list."""
    low = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in low: return low[k]
    return None

def _to_number(x, nd=2, allow_inf=False):
    try:
        if x is None: return 0.0
        if isinstance(x, (int, float, np.floating)):
            if not allow_inf and (not np.isfinite(x)): return 0.0
            return round(float(x), nd)
        return round(float(x), nd)
    except Exception:
        return 0.0

def compute_from_trades(trades_csv, initial_capital=INITIAL_CAPITAL):
    if not os.path.isfile(trades_csv): 
        eprint("INFO: trades.csv not found:", trades_csv)
        return {}

    df = pd.read_csv(trades_csv)
    if df.empty:
        eprint("INFO: trades.csv is empty")
        return {}

    # --- Detect columns ---
    pnl_col = _pick_col(df, ["pnl","profit","netpnl","net_pnl","pl"])
    if pnl_col is None:
        eprint("WARN: PnL column not found in trades.csv")
        return {}

    # Keep only closed/filled trades if a status column exists
    st_col = _pick_col(df, ["status"])
    if st_col:
        closed = df[df[st_col].astype(str).str.lower().isin(
            ["closed","filled","exit","complete","executed"]
        )].copy()
        if closed.empty:
            closed = df.copy()
    else:
        closed = df.copy()

    trades = int(len(closed))
    if trades == 0:
        equity0 = float(initial_capital)
        return {
            "trades": 0,
            "win_rate": 0.0,
            "roi_pct": 0.0,
            "final_capital": equity0,
            "max_dd_pct": 0.0,
            "time_dd_bars": 0,
            "profit_factor": 0.0,
            "rr": 0.0,
            "sharpe_ratio": 0.0,
        }

    # --- Basic stats ---
    wins   = int((closed[pnl_col] > 0).sum())
    losses = int((closed[pnl_col] < 0).sum())
    win_rate = 100.0 * wins / trades if trades else 0.0

    # Equity curve & ROI
    equity = initial_capital + closed[pnl_col].cumsum()
    final_capital = float(equity.iloc[-1])
    roi_pct = 100.0 * (final_capital - initial_capital) / initial_capital

    # Drawdown (per-trade)
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd_pct = 100.0 * float(dd.min())
    time_dd_bars = int((dd < 0).sum())

    # Profit Factor
    gross_profit = float(closed.loc[closed[pnl_col] > 0, pnl_col].sum())
    gross_loss   = float(-closed.loc[closed[pnl_col] < 0, pnl_col].sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    # R:R
    avg_win  = float(closed.loc[closed[pnl_col] > 0, pnl_col].mean()) if wins else 0.0
    avg_loss = float(-closed.loc[closed[pnl_col] < 0, pnl_col].mean()) if losses else 0.0
    if avg_loss > 0:
        rr = avg_win / avg_loss
    else:
        rr = float("inf") if avg_win > 0 else 0.0

    # Sharpe (per-trade, sqrt(N) scaled)
    rets = closed[pnl_col] / float(initial_capital)
    sharpe = (rets.mean() / rets.std()) * math.sqrt(len(rets)) if rets.std() not in (0, None) else 0.0

    # --- Return rounded metrics (readable), but keep "Inf" strings for PF/RR when applicable ---
    out = {
        "trades": trades,
        "win_rate": _to_number(win_rate, 2),
        "roi_pct": _to_number(roi_pct, 2),
        "final_capital": _to_number(final_capital, 2),
        "max_dd_pct": _to_number(max_dd_pct, 2),
        "time_dd_bars": int(time_dd_bars),
        "profit_factor": ("Inf" if not np.isfinite(profit_factor) else _to_number(profit_factor, 2)),
        "rr": ("Inf" if not np.isfinite(rr) else _to_number(rr, 2)),
        "sharpe_ratio": _to_number(sharpe, 2),
    }
    return out

def main():
    if len(sys.argv) < 2:
        print("usage: python tools/ensure_metrics.py <reports_dir>")
        sys.exit(0)

    rep_dir = sys.argv[1]
    os.makedirs(rep_dir, exist_ok=True)
    metrics_path = os.path.join(rep_dir, "metrics.json")
    trades_csv = os.path.join(rep_dir, "trades.csv")

    # start from existing metrics (if any)
    out = {}
    if os.path.isfile(metrics_path):
        try:
            out = json.load(open(metrics_path, "r", encoding="utf-8")) or {}
        except Exception as ex:
            eprint("WARN: failed reading existing metrics.json:", ex)

    fresh = compute_from_trades(trades_csv, INITIAL_CAPITAL)
    out.update(fresh)

    # sanity defaults
    out.setdefault("trades", 0)
    out.setdefault("win_rate", 0.0)
    out.setdefault("roi_pct", 0.0)
    out.setdefault("final_capital", INITIAL_CAPITAL)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"✅ metrics.json written → {metrics_path}")

if __name__ == "__main__":
    sys.exit(main())
