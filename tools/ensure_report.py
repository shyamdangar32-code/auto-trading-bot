#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ensure report artifacts:
- Normalize & augment metrics (including R:R) from metrics.json and/or trades.csv
- Rewrite metrics.json (adds both 'rr' and legacy 'R_R')
- Rewrite REPORT.md with all metrics incl. R:R
- Rewrite decision.json with accepted_metrics

Usage:
  python tools/ensure_report.py <reports_dir>
"""

import os, sys, json, math
import pandas as pd
import numpy as np

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))

def eprint(*a): print(*a, file=sys.stderr)

# ---------- Core helpers ----------

def _detect_pnl_col(df):
    cands = ["pnl","netpnl","net_pnl","pnl_rs","profit","pl","p&l"]
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in cands):
            return c
    return None

def compute_from_trades(trades_csv, initial_capital=INITIAL_CAPITAL):
    """Compute full metric set from trades.csv."""
    if not os.path.isfile(trades_csv):
        return {}
    df = pd.read_csv(trades_csv)
    if df.empty:
        return {}

    pnl_col = _detect_pnl_col(df)
    if pnl_col is None:
        eprint("WARN: PnL column not found in trades.csv")
        return {}

    if "status" in df.columns:
        closed = df[df["status"].astype(str).str.lower().isin(
            ["closed","filled","exit","complete"]
        )].copy()
        if closed.empty:
            closed = df.copy()
    else:
        closed = df.copy()

    trades = int(len(closed))
    wins = int((closed[pnl_col] > 0).sum())
    losses = int((closed[pnl_col] < 0).sum())
    win_rate = 100.0 * wins / trades if trades else 0.0

    equity = initial_capital + closed[pnl_col].cumsum()
    final_capital = float(equity.iloc[-1]) if trades else float(initial_capital)
    roi_pct = 100.0 * (final_capital - initial_capital) / initial_capital

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd_pct = 100.0 * float(dd.min()) if trades else 0.0
    time_dd_bars = int((dd < 0).sum()) if trades else 0

    total_profit = float(closed.loc[closed[pnl_col] > 0, pnl_col].sum())
    total_loss = float(-closed.loc[closed[pnl_col] < 0, pnl_col].sum())
    profit_factor_val = (total_profit / total_loss) if total_loss > 0 else (float("inf") if total_profit > 0 else 0.0)

    # --- R:R (robust & aligned with ensure_metrics.py) ---
    wins_series  = closed.loc[closed[pnl_col] > 0, pnl_col]
    loss_series  = closed.loc[closed[pnl_col] < 0, pnl_col].abs()
    avg_win  = float(wins_series.mean())  if wins_series.size  > 0 else 0.0
    avg_loss = float(loss_series.mean())  if loss_series.size > 0 else 0.0
    if avg_loss > 0:
        rr_val = avg_win / avg_loss
    else:
        rr_val = float("inf") if avg_win > 0 else 0.0  # <-- CHANGED: keep Inf when no losses but wins exist

    rets = closed[pnl_col] / float(initial_capital)
    sharpe_ratio = float((rets.mean() / (rets.std() or 1.0)) * np.sqrt(len(rets))) if trades else 0.0

    # presentational rounding (keep Inf as "Inf")
    profit_factor = ("Inf" if not np.isfinite(profit_factor_val) else round(profit_factor_val, 2))
    rr = ("Inf" if not np.isfinite(rr_val) else round(rr_val, 2))

    return {
        "trades": trades,
        "win_rate": round(win_rate, 2),
        "roi_pct": round(roi_pct, 2),
        "final_capital": round(final_capital, 2),
        "profit_factor": profit_factor,
        "rr": rr,
        "max_dd_pct": round(max_dd_pct, 2),
        "time_dd_bars": time_dd_bars,
        "sharpe_ratio": round(sharpe_ratio, 2),
    }

def normalize_keys(m):
    """Map any legacy keys to our canonical names."""
    return {
        "trades":        int(m.get("trades", m.get("n_trades", 0) or 0)),
        "win_rate":      float(m.get("win_rate", m.get("winrate", 0) or 0)),
        "roi_pct":       float(m.get("roi_pct", m.get("ROI", 0) or 0)),
        "profit_factor": m.get("profit_factor", 0),
        "rr":            (m.get("rr", m.get("R_R", 0) or 0)),  # may be "Inf" string
        "max_dd_pct":    float(m.get("max_dd_pct", m.get("max_dd_perc", 0) or 0)),
        "time_dd_bars":  int(m.get("time_dd_bars", m.get("time_dd", 0) or 0)),
        "sharpe_ratio":  float(m.get("sharpe_ratio", m.get("sharpe", 0) or 0)),
        "final_capital": float(m.get("final_capital", m.get("FinalCapital", 0) or 0)),
    }

def format_report(m):
    # tolerate "Inf" string for PF/RR
    pf = m['profit_factor']
    rr = m['rr']
    if isinstance(pf, (int, float)) and not np.isfinite(pf): pf = "Inf"
    if isinstance(rr, (int, float)) and not np.isfinite(rr): rr = "Inf"
    try_rr = f"{rr:.2f}" if isinstance(rr, (int,float)) else str(rr)
    try_pf = f"{pf:.2f}" if isinstance(pf, (int,float)) else str(pf)
    return (
        "# Backtest Summary\n\n"
        f"- **Trades**: {m['trades']}\n"
        f"- **Win-rate**: {m['win_rate']:.2f}%\n"
        f"- **ROI**: {m['roi_pct']:.2f}%\n"
        f"- **Profit Factor**: {try_pf}\n"
        f"- **R:R**: {try_rr}\n"
        f"- **Max DD**: {m['max_dd_pct']:.2f}%\n"
        f"- **Time DD (bars)**: {m['time_dd_bars']}\n"
        f"- **Sharpe**: {m['sharpe_ratio']:.2f}\n\n"
        "![Equity](equity_curve.png)\n"
        "![Drawdown](drawdown.png)\n"
    )

# ---------- Main ----------

def main():
    if len(sys.argv) < 2:
        print("usage: python tools/ensure_report.py <reports_dir>")
        return 0

    rep = sys.argv[1]
    os.makedirs(rep, exist_ok=True)

    metrics_path  = os.path.join(rep, "metrics.json")
    trades_csv    = os.path.join(rep, "trades.csv")
    report_md     = os.path.join(rep, "REPORT.md")
    decision_json = os.path.join(rep, "decision.json")

    # 1) load + normalize existing metrics (optional)
    m = {}
    if os.path.isfile(metrics_path):
        try:
            m = json.load(open(metrics_path, "r", encoding="utf-8"))
        except Exception as ex:
            eprint("WARN: metrics.json read failed:", ex)
            m = {}

    nm = normalize_keys(m)

    # 2) if critical fields missing/zero -> augment from trades.csv
    def _is_zero_or_missing(x):
        # treat "Inf" as non-zero
        if isinstance(x, str) and x.strip().lower() == "inf": return False
        try:
            return float(x) == 0.0
        except Exception:
            return True

    need_aug = (
        nm["trades"] == 0
        or _is_zero_or_missing(nm["rr"])
        or _is_zero_or_missing(nm["roi_pct"])
        or _is_zero_or_missing(nm["max_dd_pct"])
        or _is_zero_or_missing(nm["sharpe_ratio"])
    )
    if need_aug:
        aug = compute_from_trades(trades_csv)
        if aug:
            nm.update(aug)

    # 3) write back a *consistent* metrics.json (include legacy aliases too)
    write_metrics = dict(nm)
    write_metrics["R_R"] = nm["rr"]                 # legacy alias
    write_metrics["ROI"] = nm["roi_pct"]            # legacy alias
    write_metrics["max_dd_perc"] = nm["max_dd_pct"] # legacy alias
    write_metrics["FinalCapital"] = nm["final_capital"]  # legacy alias

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(write_metrics, f, indent=2, ensure_ascii=False)

    # 4) REPORT.md
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(format_report(nm))

    # 5) decision.json (simple, consistent payload)
    decision = {
        "mode": "auto",
        "applied": "accepted",
        "reason": "normalized & augmented from metrics/trades",
        "accepted_metrics": {
            "trades": nm["trades"],
            "win_rate": nm["win_rate"],
            "ROI": nm["roi_pct"],
            "profit_factor": nm["profit_factor"],
            "R_R": nm["rr"],
            "max_dd_perc": nm["max_dd_pct"],
            "time_dd_bars": nm["time_dd_bars"],
            "sharpe": nm["sharpe_ratio"]
        }
    }
    with open(decision_json, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, ensure_ascii=False)

    print("✅ report artifacts refreshed:",
          os.path.basename(metrics_path),
          os.path.basename(report_md),
          os.path.basename(decision_json))
    return 0

if __name__ == "__main__":
    sys.exit(main())
