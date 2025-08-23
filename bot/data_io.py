# bot/data_io.py
import os
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd

# Zerodha only (no Yahoo fallback)
from kiteconnect import KiteConnect


# ------------------------- helpers -------------------------

def _parse_period_days(period: str) -> int:
    """Convert lookback like '1y','6mo','30d' -> days."""
    p = period.strip().lower()
    if p.endswith("y"):
        return int(p[:-1]) * 365
    if p.endswith("mo"):
        return int(p[:-2]) * 30
    if p.endswith("d"):
        return int(p[:-1])
    return 365  # default

def _map_interval(interval: str) -> str:
    """Map our interval names to Zerodha intervals."""
    return {
        "1d": "day",
        "1h": "60minute",
        "30m": "30minute",
        "15m": "15minute",
        "5m": "5minute",
    }.get(interval, "day")

def _get_kite() -> KiteConnect:
    api_key = os.getenv("ZERODHA_API_KEY", "").strip()
    access  = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
    if not api_key or not access:
        raise RuntimeError("ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN not set")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    try:
        kite.profile()  # quick check
        print("✅ Zerodha token OK.")
    except Exception as e:
        raise RuntimeError(f"Zerodha token problem: {e}")
    return kite


# ------------------------- Zerodha download -------------------------

def zerodha_prices(instrument_token: int, period: str, interval: str) -> pd.DataFrame:
    kite = _get_kite()

    days = _parse_period_days(period)
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=days)
    tf = _map_interval(interval)

    print(f"🟢 Zerodha: token={instrument_token} {tf} {from_dt:%Y-%m-%d}->{to_dt:%Y-%m-%d}")
    candles = kite.historical_data(
        instrument_token=int(instrument_token),
        from_date=from_dt,
        to_date=to_dt,
        interval=tf,
    )
    if not candles:
        raise RuntimeError("Zerodha returned no candles")

    df = pd.DataFrame(candles)
    # standardize column names
    df.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"
    }, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Close"])
    print(f"✅ Zerodha OK: {len(df)} rows")
    return df


# ------------------------- unified entry -------------------------

def prices(
    symbol: str,
    period: str,
    interval: str,
    zerodha_enabled: bool = True,
    zerodha_instrument_token: Optional[int] = None,
) -> pd.DataFrame:
    """
    Zerodha-only pipeline. We require instrument_token when enabled.
    """
    if not zerodha_enabled:
        raise RuntimeError("Zerodha disabled in config, and Yahoo fallback is removed by request.")

    if not zerodha_instrument_token:
        raise RuntimeError("Missing `zerodha_instrument_token` in config.yaml")

    return zerodha_prices(int(zerodha_instrument_token), period, interval)
