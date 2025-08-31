#!/usr/bin/env python3
# runner_intraday_options.py

import os
import argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.broker.zerodha import ZerodhaClient
from src.strategy.signals import prepare_signals
from src.utils.telegram import send_telegram_message

IST = timezone(timedelta(hours=5, minutes=30))

# ---------------- utils ----------------

def ist_now():
    return datetime.now(IST)

def ensure_datetime_index(df):
    """Ensure candle dataframe has proper datetime index with tzinfo (IST)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    # Convert UTC → IST
    df.index = df.index.tz_convert(IST)
    return df

# ---------------- runner ----------------

def run_intraday(config):
    # init broker
    client = ZerodhaClient(
        api_key=config.api_key,
        api_secret=config.api_secret,
        access_token=config.access_token,
        paper=True
    )

    # download candles
    print("⬇️ Downloading index candles…")
    df = client.download_index("NIFTY", interval="5minute", lookback_days=30)

    if df is None or len(df) == 0:
        raise ValueError("No data downloaded from Zerodha.")

    print(f"✅ Zerodha OK: {len(df)} rows")
    df = ensure_datetime_index(df)

    # run strategy signals
    print("🧮 Running intraday backtest (index-level)…")
    signals = prepare_signals(df)

    # summary
    print(f"✅ Generated {len(signals)} signals")
    send_telegram_message(f"Intraday run complete ✅\nSignals: {len(signals)}")

    return signals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # load config
    import yaml
    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    config = SimpleNamespace(**raw)

    run_intraday(config)
