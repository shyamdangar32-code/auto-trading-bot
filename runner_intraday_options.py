#!/usr/bin/env python3
# runner_intraday_options.py

import os, sys, json, argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))
def ist_now():
    return datetime.now(IST)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Example signals (replace with live logic)
    signals = [{"side": "BUY", "pnl": 500}, {"side": "SELL", "pnl": -200}, {"side": "BUY", "pnl": 300}]
    trades = signals
    pnl_series = pd.Series([t["pnl"] for t in trades]).cumsum()

    total_pnl = pnl_series.iloc[-1]
    win_rate = round(100 * sum(1 for t in trades if t["pnl"] > 0) / len(trades), 2)
    roi = round(total_pnl / 100000 * 100, 2)  # assume 1L capital

    # Save metrics.json
    summary = {
        "n_signals": len(signals),
        "n_trades": len(trades),
        "pnl": total_pnl,
        "win_rate": win_rate,
        "roi_pct": roi,
        "config": {
            "capital": 100000,
            "order_qty": 1,
            "datetime": str(ist_now()),
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ----- Charts -----
    # Equity curve
    plt.figure(figsize=(8, 4))
    plt.plot(pnl_series, label="Equity Curve", color="blue")
    plt.title("Equity Curve (Intraday Paper)")
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative P&L")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "equity_curve.png"))
    plt.close()

    # Drawdown curve
    cummax = pnl_series.cummax()
    drawdown = pnl_series - cummax
    plt.figure(figsize=(8, 4))
    plt.plot(drawdown, color="red", label="Drawdown")
    plt.title("Drawdown (Intraday Paper)")
    plt.xlabel("Trade #")
    plt.ylabel("P&L vs Peak")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "drawdown.png"))
    plt.close()

    print("✅ Intraday (paper) run finished. metrics.json + charts saved.")

if __name__ == "__main__":
    main()
