# tools/run_backtest.py
# Driver that fetches OHLC from Kite and calls your bot/backtest.py

from __future__ import annotations
import os, argparse
from datetime import datetime, timedelta, date, time

import pandas as pd
from kiteconnect import KiteConnect

from bot.backtest import run_backtest, save_reports
# your strategy/indicators are used inside run_backtest via prepare_signals()

def ist_day_bounds(d: date):
    # Zerodha intraday is IST; naive datetimes are OK on GA runners
    start = datetime(d.year, d.month, d.day, 9, 15, 0)
    end   = datetime(d.year, d.month, d.day, 15, 30, 0)
    return start, end

def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def fetch_range_ohlc(kite: KiteConnect, token: int, d0: date, d1: date, interval: str) -> pd.DataFrame:
    """Loop by trading day to avoid window limits; concatenate to one DataFrame."""
    frames = []
    for d in daterange(d0, d1):
        start, end = ist_day_bounds(d)
        try:
            raw = kite.historical_data(token, start, end, interval=interval)
        except Exception as e:
            print(f"⚠️  Skip {d}: {e}")
            continue
        if not raw:
            continue
        df = pd.DataFrame(raw)
        # Normalize columns to what your backtester expects
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        frames.append(df[["datetime", "Open", "High", "Low", "Close", "Volume"]])
    if not frames:
        return pd.DataFrame(columns=["datetime","Open","High","Low","Close","Volume"]).set_index("datetime")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("datetime").drop_duplicates("datetime")
    out = out.set_index("datetime")
    return out

def load_cfg():
    # Use your existing helper if present (keeps a single source of truth)
    try:
        from bot.config import load_config  # type: ignore
        return load_config()
    except Exception:
        import yaml
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token", type=int, required=True, help="Instrument token")
    p.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    p.add_argument("--interval", default="5minute", choices=["3minute","5minute","10minute","15minute"])
    p.add_argument("--out_dir", default="reports/backtest")
    args = p.parse_args()

    api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise SystemExit("Missing ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN envs")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)

    d0 = datetime.fromisoformat(args.date_from).date()
    d1 = datetime.fromisoformat(args.date_to).date()

    print(f"📥 Fetching {args.interval} candles for token {args.token} {d0} → {d1} …")
    prices = fetch_range_ohlc(kite, args.token, d0, d1, args.interval)
    if prices.empty:
        raise SystemExit("No candles fetched for the chosen range.")

    cfg = load_cfg()
    print("⚙️  Using trade config:", {
        k: cfg.get(k) for k in ["order_qty","capital_rs","reentry_max","reentry_cooldown"]
        if k in cfg
    })

    summary, trades_df, equity_ser = run_backtest(prices, cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    save_reports(args.out_dir, summary, trades_df, equity_ser)

    print("✅ Backtest complete")
    print("   Summary:", summary)
    print(f"   Files written in: {args.out_dir}")

if __name__ == "__main__":
    main()
