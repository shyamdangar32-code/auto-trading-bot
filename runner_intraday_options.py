#!/usr/bin/env python3
# runner_intraday_options.py

import os
import sys
import pathlib
from datetime import datetime, date, timedelta, timezone
import pandas as pd

# ---------------- path fixes ----------------
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "core_backtest"))

# ---------------- safe imports ----------------
try:
    from utils import data_io
except Exception:
    try:
        import data_io
    except Exception as e:
        data_io = None
        print(f"WARN: data_io import failed → {e!r}")

try:
    from utils.telegram import TelegramNotifier
except Exception:
    TelegramNotifier = None
    print("WARN: TelegramNotifier import failed")

# ---------------- timezone helpers ----------------
IST = timezone(timedelta(hours=5, minutes=30))

def ist_now():
    return datetime.now(IST)

def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Sat=5, Sun=6

# ---------------- safe candle fetch ----------------
def get_index_candles_safe(symbol, interval, start, end):
    if (data_io is not None) and hasattr(data_io, "get_index_candles"):
        try:
            return data_io.get_index_candles(symbol, interval, start, end)
        except Exception as e:
            print(f"WARN: data_io.get_index_candles error → {e!r}")
    print("WARN: data_io.get_index_candles unavailable → returning empty DataFrame")
    return pd.DataFrame(columns=["date","open","high","low","close","volume"])

# ---------------- main runner ----------------
def main():
    symbol = "NIFTY"
    interval = "5minute"
    start = date.today() - timedelta(days=30)
    end = date.today()

    df_idx = get_index_candles_safe(symbol, interval, start, end)

    if df_idx.empty:
        print("No data frame available; wrote minimal reports.")
        return

    # TODO: your strategy logic here
    print(df_idx.head())

    # Telegram notify
    if TelegramNotifier:
        notifier = TelegramNotifier()
        notifier.send_message("Runner finished with data available.")
    else:
        print("TelegramNotifier unavailable → skipped message.")

if __name__ == "__main__":
    main()
