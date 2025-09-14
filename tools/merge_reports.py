# tools/merge_reports.py
import argparse, json, os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_metrics(p):
    with open(p, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing day folders (YYYY-MM-DD)")
    ap.add_argument("--out", required=True, help="output folder for merged report")
    ap.add_argument("--capital_rs", required=True, type=float)
    args = ap.parse_args()

    days = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root,d))])
    assert days, "No per-day folders found to merge."

    os.makedirs(args.out, exist_ok=True)

    # --- Merge trades.csv (if present) ---
    trades_list = []
    for d in days:
        p = os.path.join(args.root, d, "trades.csv")
        if os.path.exists(p):
            t = pd.read_csv(p)
            trades_list.append(t)
    trades = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    if not trades.empty and "timestamp" in trades.columns:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        trades = trades.sort_values("timestamp").reset_index(drop=True)
        trades.to_csv(os.path.join(args.out, "trades.csv"), index=False)

    # --- Merge equity series ---
    eq_list = []
    offset = 0.0
    merged = []
    for d in days:
        p = os.path.join(args.root, d, "equity.csv")
        if not os.path.exists(p):
            continue
        e = pd.read_csv(p)
        # expect columns: timestamp, equity (or 'Equity')
        col = "equity" if "equity" in e.columns else ("Equity" if "Equity" in e.columns else None)
        tcol = "timestamp" if "timestamp" in e.columns else ("Date" if "Date" in e.columns else None)
        if col is None or tcol is None: 
            continue
        e[tcol] = pd.to_datetime(e[tcol])
        e = e.sort_values(tcol)
        base = e[col].iloc[0]
        e[col] = e[col] - base + (args.capital_rs + offset)
        offset = e[col].iloc[-1] - args.capital_rs  # carry forward PnL
        merged.append(e[[tcol, col]])

    equity = pd.concat(merged, ignore_index=True) if merged else pd.DataFrame(columns=["timestamp","equity"])
    if not equity.empty:
        equity.rename(columns={equity.columns[0]:"timestamp", equity.columns[1]:"equity"}, inplace=True)
        equity.to_csv(os.path.join(args.out, "equity.csv"), index=False)

        # Drawdown
        eq = equity["equity"].values
        roll_max = np.maximum.accumulate(eq)
        dd = eq - roll_max
        dd_df = pd.DataFrame({"timestamp": equity["timestamp"], "drawdown": dd})

        # Plots
        plt.figure(figsize=(12,4))
        plt.plot(equity["timestamp"], equity["equity"])
        plt.title("Equity Curve")
        plt.xlabel("Date"); plt.tight_layout()
        plt.savefig(os.path.join(args.out, "equity_curve.png")); plt.close()

        plt.figure(figsize=(12,3))
        plt.plot(dd_df["timestamp"], dd_df["drawdown"])
        plt.title("Drawdown")
        plt.xlabel("Date"); plt.tight_layout()
        plt.savefig(os.path.join(args.out, "drawdown.png")); plt.close()

        # Metrics (merged)
        start_equity = equity["equity"].iloc[0]
        end_equity   = equity["equity"].iloc[-1]
        roi = (end_equity/start_equity - 1.0) * 100.0

        # win/loss from trades if available
        win_rate = np.nan
        profit_factor = np.nan
        trades_cnt = 0
        if not trades.empty and "pnl" in trades.columns:
            pnl = trades["pnl"].astype(float)
            trades_cnt = len(pnl)
            wins = (pnl > 0).sum()
            win_rate = (wins / trades_cnt) * 100.0 if trades_cnt else np.nan
            gp = pnl[pnl>0].sum()
            gl = -pnl[pnl<0].sum()
            profit_factor = (gp / gl) if gl>0 else np.nan

        # sharpe approx using minute returns
        ret = equity["equity"].pct_change().dropna()
        sharpe = (ret.mean()/ret.std()*np.sqrt(252*6.5*60)) if ret.std() > 0 else np.nan  # rough

        max_dd_perc = (dd.min()/roll_max.max())*100.0 if len(dd)>0 else np.nan
        time_dd_bars = int((dd<0).sum())

        metrics = {
            "trades": trades_cnt,
            "win_rate": round(float(win_rate), 2) if win_rate==win_rate else 0.0,
            "ROI": round(float(roi), 2),
            "profit_factor": round(float(profit_factor), 2) if profit_factor==profit_factor else 0.0,
            "R_R": 0.0,
            "max_dd_perc": round(float(max_dd_perc), 2) if max_dd_perc==max_dd_perc else 0.0,
            "time_dd_bars": time_dd_bars,
            "sharpe": round(float(sharpe), 2) if sharpe==sharpe else 0.0
        }
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(args.out, "REPORT.md"), "w") as f:
            f.write("# Backtest Summary (Merged)\n\n")
            for k,v in metrics.items():
                f.write(f"- **{k.replace('_',' ').title()}**: {v}\n")
            f.write("\n![Equity](equity_curve.png)\n\n![Drawdown](drawdown.png)\n")

if __name__ == "__main__":
    main()
