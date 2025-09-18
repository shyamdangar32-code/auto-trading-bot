#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build/refresh metrics.json for a backtest reports directory.

Reads trades.csv to compute:
- trades, win_rate, roi_pct, final_capital
- max_dd_pct, time_dd_bars
- profit_factor, rr (avg win / avg loss)
- sharpe_ratio (per-trade; sqrt(N) scaled)

Writes/updates `metrics.json` in the same reports dir.
"""

import os, sys, json, math
import pandas as pd
import numpy as np

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))

def eprint(*a): print(*a, file=sys.stderr)

def load_existing(path):
    if os.path.isfile(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception as ex:
            eprint("WARN: failed to read existing metrics.json:", ex)
    return {}

def compute_from_trades(trades_csv, initial_capital=INITIAL_CAPITAL):
    if not os.path.isfile(trades_csv):
        return {}

    df = pd.read_csv(trades_csv)
    if df.empty:
        return {}

    # robust PnL column detection
    candidates = ["pnl", "netpnl", "net_pnl", "pnl_rs", "profit", "pl", "p&l"]
    pnl_col = next((c for c in df.columns if any(k in c.lower() for k in candidates)), None)
    if pnl_col is None:
        eprint("WARN: PnL column not found in trades.csv")
        return {}

    # keep only closed/filled trades if a status column exists
    if "status" in df.columns:
        closed = df[df["status"].astype(str).str.lower().isin(
            ["closed", "filled", "exit", "complete"]
        )].copy()
        if closed.empty:
            closed = df.copy()
    else:
        closed = df.copy()

    trades = int(len(closed))
    wins = int((closed[pnl_col] > 0).sum())
    losses = int((closed[pnl_col] < 0).sum())
    win_rate = 100.0 * wins / trades if trades else 0.0

    # equity curve
    equity = initial_capital + closed[pnl_col].cumsum()
    final_capital = float(equity.iloc[-1]) if trades else float(initial_capital)
    roi_pct = 100.0 * (final_capital - initial_capital) / initial_capital

    # drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd_pct = 100.0 * float(dd.min()) if trades else 0.0
    time_dd_bars = int((dd < 0).sum()) if trades else 0

    # profit factor
    gross_profit = float(closed.loc[closed[pnl_col] > 0, pnl_col].sum())
    gross_loss = float(-closed.loc[closed[pnl_col] < 0, pnl_col].sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    # ----- R:R (FIXED) -----
    # Use absolute loss magnitude to avoid sign/NaN issues
    wins_series  = closed.loc[closed[pnl_col] > 0, pnl_col]
    loss_series  = closed.loc[closed[pnl_col] < 0, pnl_col].abs()
    avg_win  = float(wins_series.mean())  if wins_series.size  > 0 else 0.0
    avg_loss = float(loss_series.mean())  if loss_series.size > 0 else 0.0
    rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    # Sharpe (per-trade). Annualization as sqrt(N) over trade count.
    rets = closed[pnl_col] / float(initial_capital)
    sharpe_ratio = float((rets.mean() / (rets.std() or 1.0)) * math.sqrt(len(rets))) if trades else 0.0

    return {
        "trades": trades,
        "win_rate": round(win_rate, 2),
        "roi_pct": round(roi_pct, 2),
        "final_capital": round(final_capital, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "time_dd_bars": time_dd_bars,
        "profit_factor": (round(profit_factor, 2) if np.isfinite(profit_factor) else "Inf"),
        "rr": round(rr, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
    }

def main():
    if len(sys.argv) < 2:
        print("usage: python tools/ensure_metrics.py <reports_dir>")
        sys.exit(0)

    rep_dir = sys.argv[1]
    os.makedirs(rep_dir, exist_ok=True)

    metrics_path = os.path.join(rep_dir, "metrics.json")
    trades_csv = os.path.join(rep_dir, "trades.csv")

    out = load_existing(metrics_path)
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
