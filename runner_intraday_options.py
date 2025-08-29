#!/usr/bin/env python3
# runner_intraday_options.py

from __future__ import annotations
import os, json, argparse
import pandas as pd
from datetime import datetime, timezone, timedelta

from bot.config import load_config, debug_fingerprint
from bot.data_io import prices as fetch_prices
from bot.backtest import run_backtest, save_reports

IST = timezone(timedelta(hours=5, minutes=30))
today_ist = lambda: datetime.now(IST).date()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--lookback_days", type=int, default=5, help="How many calendar days of candles to pull")
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    intr = cfg.get("intraday_options", {}) or {}

    # Core params
    interval   = intr.get("timeframe", "5minute")
    token      = intr.get("index_token")  # Zerodha instrument token for NIFTY/BANKNIFTY index
    capital    = float(intr.get("capital_rs", cfg.get("capital_rs", 100000)))
    order_qty  = int(intr.get("order_qty", cfg.get("order_qty", 1)))

    # Strategy knobs (fall back to top level if needed)
    strat_cfg = {
        "ema_fast":       intr.get("ema_fast", 21),
        "ema_slow":       intr.get("ema_slow", 50),
        "rsi_len":        intr.get("rsi_len", 14),
        "rsi_buy":        intr.get("rsi_buy", 30),
        "rsi_sell":       intr.get("rsi_sell", 70),
        "adx_len":        intr.get("adx_len", 14),
        "atr_len":        intr.get("atr_len", 14),
        "stop_atr_mult":  intr.get("sl_atr_mult", intr.get("stop_atr_mult", 2.0)),
        "take_atr_mult":  intr.get("tgt_rr", intr.get("take_atr_mult", 3.0)),
        "trailing_enabled": True,
        "trail_start_atr": intr.get("trail_start_atr", 1.0),
        "trail_atr_mult":  intr.get("trail_atr_mult", 1.0),
        "reentry_max":     intr.get("reentry_max", 2),
        "reentry_cooldown":intr.get("reentry_cooldown", 3),
        "order_qty":       order_qty,
        "capital_rs":      capital,
    }

    # Pull last N days to have warmup for indicators
    period_str = f"{max(args.lookback_days, 3)}d"
    if not token:
        raise SystemExit("intraday_options.index_token missing in config.yaml")
    df = fetch_prices(symbol="", period=period_str, interval=interval,
                      zerodha_enabled=True, zerodha_instrument_token=int(token))

    # Filter to today only (intraday)
    d0 = df[df["Date"].dt.tz_convert("Asia/Kolkata").dt.date == today_ist()]
    # If today has no candles yet (e.g., holiday / late night), fall back to last session
    if d0.empty:
        d0 = df.iloc[-len(df)//args.lookback_days:]  # last chunk as a best-effort

    # Backtest engine over the selected slice
    d0 = d0.rename(columns={"Date": "timestamp"}).set_index("timestamp")
    d0 = d0[["Open","High","Low","Close"]]
    summary, trades, equity = run_backtest(d0, strat_cfg)

    # Write artifacts
    save_reports(args.out_dir, summary, trades, equity)

    print("✅ Intraday engine finished.")
    print("Artifacts in:", args.out_dir)

if __name__ == "__main__":
    main()
