#!/usr/bin/env python3
# runner_intraday_options.py
# ==========================
# Intraday Options Algo Trading Runner (BANKNIFTY/NIFTY)
# Handles live/paper trading, entry/exit, SL/target, trailing, re-entry

import os
import sys
import yaml
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from kiteconnect import KiteConnect
from bot.strategy import IntradayOptionsStrategy
from bot.execution import ExecutionEngine
from bot.data_io import get_index_data
from bot.evaluation import save_daily_metrics
from tools.ensure_report import ensure_report_dir
from tools.ensure_metrics import ensure_metrics_file
from bot.utils import ist_now, parse_time, logger

# Load config
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

IST = timezone(timedelta(hours=5, minutes=30))

def main():
    ensure_report_dir()
    ensure_metrics_file()

    # --- Setup Zerodha (paper or live) ---
    kite = None
    if CONFIG["zerodha_enabled"] and CONFIG["intraday_options"]["live_trading"]:
        kite = KiteConnect(api_key=CONFIG["zerodha_api_key"])
        kite.set_access_token(CONFIG["zerodha_access_token"])

    # --- Init strategy + execution engine ---
    strategy = IntradayOptionsStrategy(CONFIG)
    executor = ExecutionEngine(CONFIG, kite)

    # --- Trading window ---
    start = parse_time(CONFIG["intraday_options"]["start_time"])
    end = parse_time(CONFIG["intraday_options"]["end_time"])

    logger.info("Intraday Options Runner started.")

    while True:
        now = ist_now().time()
        if now < start:
            logger.info("⏳ Waiting for market open...")
            time.sleep(30)
            continue
        if now >= end:
            logger.info("✅ Trading window ended. Closing positions.")
            executor.square_off_all()
            break

        # --- Fetch latest candle data ---
        df = get_index_data(CONFIG)
        if df is None or len(df) < CONFIG["intraday_options"]["ma_len"]:
            time.sleep(10)
            continue

        # --- Generate signals (Entry / Exit / Re-entry / Trailing) ---
        signals = strategy.generate_signals(df)

        # --- Execute signals ---
        for sig in signals:
            executor.handle_signal(sig)

        # --- Update metrics ---
        save_daily_metrics(executor.trades, CONFIG)

        time.sleep(60)  # Run every candle

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("⚠️ Stopped manually.")
    except Exception as e:
        logger.exception(f"❌ Runner crashed: {e}")
