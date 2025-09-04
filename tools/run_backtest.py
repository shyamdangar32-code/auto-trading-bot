#!/usr/bin/env python3
# tools/run_backtest.py

from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# repo local imports
from bot.pricefeed import get_zerodha_ohlc  # your existing helper
from bot.backtest import run_backtest       # NOTE: save_reports is now local here


# ------------------------
# Local helper: save all artifacts (no import from bot.backtest)
# ------------------------
def save_reports(out_dir: str,
                 profile: str,
                 summary: dict,
                 trades_df: pd.DataFrame,
                 equity_ser: pd.Series) -> None:
    """
    Write: report.md, metrics.json, equity.csv, drawdown/equity charts.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- metrics.json
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- equity.csv
    if not isinstance(equity_ser, pd.Series):
        equity_ser = pd.Series(dtype=float)
    equity_path = out / "equity.csv"
    equity_ser.to_csv(equity_path, header=["equity"], index_label="ts")

    # ---- charts
    # Equity
    plt.figure()
    equity_ser.reset_index(drop=True).plot()
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity (₹)")
    plt.tight_layout()
    plt.savefig(out / "equity_curve.png")
    plt.close()

    # Drawdown (simple peak-to-valley from equity)
    if len(equity_ser) > 0:
        roll_max = equity_ser.cummax()
        dd = equity_ser - roll_max
    else:
        dd = pd.Series(dtype=float)

    plt.figure()
    dd.reset_index(drop=True).plot()
    plt.title("Drawdown (₹)")
    plt.xlabel("Trade #")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out / "drawdown.png")
    plt.close()

    # ---- report.md (short human summary)
    report_md = out / "report.md"
    lines = [
        f"Backtest Summary (SUCCESS) — profile: {profile}",
        f"Underlying: {summary.get('underlying','NIFTY')}   Interval: {summary.get('interval','1m')}",
        f"Period: {summary.get('period_start','?')} -> {summary.get('period_end','?')}",
        f"Trades: {summary.get('trades',0)}   Win-rate: {summary.get('win_rate_pct',0):.2f}%",
        f"ROI: {summary.get('roi_pct',0):.2f}%   PF: {summary.get('pf',0):.2f}",
        f"R:R: {summary.get('rr',0):.2f}   Sharpe: {summary.get('sharpe',0):.02f}",
        f"Max DD: {summary.get('max_dd_pct',0):.02f}%   Time DD: {summary.get('time_dd_bars',0)} bars",
        f"Bars: {summary.get('bars',0)}   ATR-bars: {summary.get('atr_bars',0)}",
        f"Setups: long={summary.get('setups_long',0)} short={summary.get('setups_short',0)}   Trades taken: {summary.get('trades',0)}",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")


# ------------------------
# CLI
# ------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--token", required=True, type=int, help="Zerodha instrument token")
    p.add_argument("--interval", default="minute")
    p.add_argument("--profile", default="loose", choices=["loose", "medium", "strict"])
    p.add_argument("--use-block", default="", help="label for profile/block in logs")
    p.add_argument("--cfg", default="config.yaml")
    p.add_argument("--out", default="reports")  # directory to write artifacts
    return p.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    cfg = load_config(args.cfg)

    # Fetch prices (you already have this util in your project)
    print(f"📅 Fetching Zerodha OHLC: token={args.token} interval={args.interval} {args.start} -> {args.end}")
    prices = get_zerodha_ohlc(
        token=args.token,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )

    # Run the backtest
    print(f"🧩 Trade config: {{'order_qty': {cfg.get('order_qty',1)}, 'capital_rs': {cfg.get('capital_rs',100000.0)} }}")
    if args.use_block:
        print(f"🧱 Using profile/block: {args.use_block}")

    summary, trades_df, equity_ser = run_backtest(prices, cfg, profile=args.profile, use_block=args.use_block)

    # Persist artifacts
    out_dir = os.path.join(args.out, args.profile)
    save_reports(out_dir, args.profile, summary, trades_df, equity_ser)

    # Also print a tiny JSON line for comparison job (keep it super small)
    comp_min = {
        "profile": args.profile,
        "trades": int(summary.get("trades", 0)),
        "win%": round(float(summary.get("win_rate_pct", 0.0)), 2),
        "PF": round(float(summary.get("pf", 0.0)), 2),
        "R:R": round(float(summary.get("rr", 0.0)), 2),
        "ROI%": round(float(summary.get("roi_pct", 0.0)), 2),
        "DD%": round(float(summary.get("max_dd_pct", 0.0)), 2),
    }
    print("==BACKTEST_COMPACT==", json.dumps(comp_min))


if __name__ == "__main__":
    main()
