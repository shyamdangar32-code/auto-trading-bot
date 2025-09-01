# tools/run_backtest.py
# -*- coding: utf-8 -*-
"""
Launch real backtest from GitHub Actions (or local),
save reports to ./reports and a compact summary to ./logs/summary.txt

CLI:
  python tools/run_backtest.py \
    --symbol BANKNIFTY --start 2025-07-01 --end 2025-08-01 \
    --interval 5m --outdir ./reports \
    --capital_rs 100000 --order_qty 1 \
    --slippage_bps 0 --broker_flat 0 --broker_pct 0 \
    --session_start "" --session_end "" --max_trades_per_day 0 \
    --extra "{}"
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd

# our package
from bot import data_io
from bot import backtest as backtest_mod
from tools.ensure_report import ensure_report  # keeps telegram step happy


def _mk_dir(p: str) -> Path:
    d = Path(p).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, help="NIFTY or BANKNIFTY")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--interval", default="5m", help="5m/10m/15m…")
    p.add_argument("--outdir", default="./reports")
    p.add_argument("--capital_rs", type=float, default=100000)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--slippage_bps", type=float, default=0.0)
    p.add_argument("--broker_flat", type=float, default=0.0)
    p.add_argument("--broker_pct", type=float, default=0.0)
    p.add_argument("--session_start", default="")
    p.add_argument("--session_end", default="")
    p.add_argument("--max_trades_per_day", type=int, default=0)
    p.add_argument("--extra", default="{}",
                   help='JSON string for future flags, e.g. \'{"signal_mode":"balanced"}\'')
    return p.parse_args()


def _none_if_blank(x: str):
    x = (x or "").strip()
    return None if x == "" else x


def main():
    args = parse_args()
    outdir = _mk_dir(args.outdir)
    _mk_dir("./logs")

    # 1) Download/index data via Zerodha (uses your active token)
    #    for index-level candles; falls back to minimal if API truly fails.
    try:
        # data_io has helpers to fetch index candles by symbol + dates
        prices = data_io.get_index_candles(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
        )
        if prices is None or len(prices) == 0:
            raise RuntimeError("Empty price dataframe")
    except Exception as e:
        # write minimal report & exit nicely, so workflow doesn't crash
        print(f"WARN: data_io.get_index_candles unavailable or returned empty -> {e}")
        ensure_report(outdir=str(outdir), note="No data; wrote minimal reports.")
        return

    # 2) Build compact config for backtest engine
    cfg = {
        "capital_rs": float(args.capital_rs),
        "order_qty": int(args.order_qty),
        "slippage_bps": float(args.slippage_bps),
        "broker_flat": float(args.broker_flat),
        "broker_pct": float(args.broker_pct),
        "session_start": _none_if_blank(args.session_start),
        "session_end": _none_if_blank(args.session_end),
        "max_trades_per_day": int(args.max_trades_per_day),
        "symbol": args.symbol,
        "interval": args.interval,
    }
    try:
        extra = json.loads(args.extra) if args.extra else {}
        if isinstance(extra, dict):
            cfg.update(extra)
    except Exception:
        pass

    # 3) Run real backtest
    try:
        # expected to return dict with trades/metrics and possibly per-trade dataframe
        result = backtest_mod.run_backtest(
            prices=prices,
            config=cfg,
        )
    except TypeError as te:
        # signature mismatch → surface clearly
        print(f"ERROR: real backtest failed -> {repr(te)}")
        ensure_report(outdir=str(outdir), note=f"Backtest error: {te}")
        return
    except Exception as e:
        print(f"ERROR: real backtest crashed -> {repr(e)}")
        ensure_report(outdir=str(outdir), note=f"Backtest error: {e}")
        return

    # 4) Persist outputs
    #    - trades.csv
    #    - metrics.json
    #    - summary.txt (for logs + telegram)
    trades_df: pd.DataFrame = result.get("trades", pd.DataFrame())
    metrics: dict = result.get("metrics", {})

    # save trades if available
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        trades_df.to_csv(outdir / "trades.csv", index=False)

    # save metrics
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # human summary
    summary_lines = [
        "Backtest Summary",
        f"Symbol: {args.symbol}",
        f"Interval: {args.interval}",
        f"Period: {args.start}..{args.end}",
        f"Trades: {int(metrics.get('trades', 0))}",
        f"Win-rate: {round(float(metrics.get('win_rate', 0.0))*100, 2)}%",
        f"ROI: {round(float(metrics.get('roi_pct', 0.0)), 2)}%",
        f"Profit Factor: {round(float(metrics.get('profit_factor', 0.0)), 2)}",
        f"R:R: {round(float(metrics.get('rr', 0.0)), 2)}",
        f"Max DD: {round(float(metrics.get('max_dd_pct', 0.0)), 2)}%",
        f"Time DD (bars): {int(metrics.get('time_dd_bars', 0))}",
        f"Sharpe: {round(float(metrics.get('sharpe', 0.0)), 2)}",
    ]
    with open("./logs/summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("✅ backtest finished; reports written.")


if __name__ == "__main__":
    main()
