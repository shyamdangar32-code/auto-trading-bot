# bot/metrics.py
# Reads reports/backtest/trades.csv and equity_curve.csv,
# computes ROI, Max Drawdown, Accuracy, R:R (median), Time drawdown (days),
# and updates/prints reports/backtest/summary.json.

import json, os
import pandas as pd
import numpy as np

def max_drawdown(series: pd.Series):
    rollmax = series.cummax()
    dd = series - rollmax
    return float(dd.min())

def time_underwater_days(series: pd.Series):
    # consecutive days below the running max
    rollmax = series.cummax()
    underwater = series < rollmax
    longest = curr = 0
    for u in underwater:
        curr = curr + 1 if u else 0
        longest = max(longest, curr)
    return int(longest)

def main():
    trades_path = "reports/backtest/trades.csv"
    equity_path = "reports/backtest/equity_curve.csv"
    summary_path = "reports/backtest/summary.json"

    if not (os.path.exists(trades_path) and os.path.exists(equity_path)):
        print("No backtest outputs found.")
        return

    tdf = pd.read_csv(trades_path)
    edf = pd.read_csv(equity_path)

    roi = float(edf["equity"].iloc[-1]) if not edf.empty else 0.0
    wins = (tdf["pnl"] > 0).sum()
    total = max(1, len(tdf))
    acc = 100.0 * wins / total
    rr_med = float(np.nanmedian(tdf.get("rr", pd.Series(dtype=float))))

    mdd = max_drawdown(edf["equity"]) if not edf.empty else 0.0
    tdd = time_underwater_days(edf["equity"]) if not edf.empty else 0

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    summary.update({
        "metrics": {
            "ROI": round(roi, 2),
            "Accuracy_pct": round(acc, 2),
            "Median_R_multiple": None if np.isnan(rr_med) else round(rr_med, 3),
            "Max_Drawdown": round(mdd, 2),
            "Time_Drawdown_days": tdd,
        }
    })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("📊 Metrics:",
          f"ROI={summary['metrics']['ROI']},",
          f"Acc={summary['metrics']['Accuracy_pct']}%,",
          f"MDD={summary['metrics']['Max_Drawdown']},",
          f"TDD={summary['metrics']['Time_Drawdown_days']}d")

if __name__ == "__main__":
    main()
