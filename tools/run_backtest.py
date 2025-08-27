#!/usr/bin/env python3
# tools/run_backtest.py

import os, sys, json, argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--interval", required=True)
    ap.add_argument("--capital_rs", type=float, default=100000)
    ap.add_argument("--order_qty", type=int, default=1)
    ap.add_argument("--mode", default="offline")
    ap.add_argument("--extra_params", default="{}")
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()

    # Demo backtest result (replace with your logic)
    trades = [{"pnl": 1200}, {"pnl": -500}, {"pnl": 800}]
    pnl_series = pd.Series([t["pnl"] for t in trades]).cumsum()

    results = {
        "trades": trades,
        "roi_pct": 1.5,
        "win_rate": 66.6,
        "profit_factor": 1.8,
        "rr": 1.4,
        "max_dd_pct": -3.2,
        "time_dd_bars": 15,
        "sharpe_ratio": 1.2,
    }

    config = vars(args)
    os.makedirs(args.out_dir, exist_ok=True)

    # Save metrics.json
    summary = {
        "n_trades": len(trades),
        "roi_pct": results["roi_pct"],
        "win_rate": results["win_rate"],
        "profit_factor": results["profit_factor"],
        "rr": results["rr"],
        "max_dd_pct": results["max_dd_pct"],
        "time_dd_bars": results["time_dd_bars"],
        "sharpe_ratio": results["sharpe_ratio"],
        "config": config,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ----- Charts -----
    # Equity curve
    plt.figure(figsize=(8, 4))
    plt.plot(pnl_series, label="Equity Curve")
    plt.title("Equity Curve")
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
    plt.title("Drawdown")
    plt.xlabel("Trade #")
    plt.ylabel("P&L vs Peak")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "drawdown.png"))
    plt.close()

    print("✅ Backtest finished. metrics.json + charts saved.")

if __name__ == "__main__":
    main()
