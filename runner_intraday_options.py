#!/usr/bin/env python3
# runner_intraday_options.py

import os
import json
import math
import argparse
from types import SimpleNamespace
from datetime import datetime, date, time, timedelta, timezone

import pandas as pd
from kiteconnect import KiteConnect

# ---------------- utils ----------------

IST = timezone(timedelta(hours=5, minutes=30))

def ensure_date(x):
    return x.date() if isinstance(x, datetime) else x

def ist_now():
    return datetime.now(IST)

def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Sat=5, Sun=6

def prev_trading_day(d: date) -> date:
    while True:
        d -= timedelta(days=1)
        if not is_weekend(d):
            return d

# ---------------- imports ----------------
# 🔹 અહીંથી "src." કાઢી નાખ્યું
from broker.zerodha import ZerodhaClient
from strategies.intraday_rsi import IntradayRSIStrategy
from utils.telegram import TelegramNotifier
from utils.report import save_report

# ---------------- main ----------------

def run_intraday(config):
    client = ZerodhaClient(
        api_key=config.api_key,
        access_token=config.access_token,
    )

    notifier = TelegramNotifier(config.tg_token, config.tg_chat)

    # Download index data
    symbol = config.symbol
    interval = config.interval
    start = config.start_date
    end = config.end_date

    print(f"📥 Downloading {symbol} {interval} data {start}→{end} ...")
    df = client.download_index(symbol, interval, start, end)

    if df.empty:
        print("⚠️ No data downloaded.")
        return

    print(f"✅ Got {len(df)} rows.")

    # Run strategy
    strat = IntradayRSIStrategy(
        rsi_period=config.rsi_period,
        ema_period=config.ema_period,
        sl=config.stop_loss,
        tp=config.take_profit,
    )
    trades, summary = strat.run(df)

    # Save report
    report_file = save_report(trades, summary, df, config.output_dir)

    # Notify
    notifier.send_message(f"Intraday run complete ✅\n{summary}")
    notifier.send_file(report_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    config = SimpleNamespace(**config_dict)

    os.makedirs(config.output_dir, exist_ok=True)

    run_intraday(config)
