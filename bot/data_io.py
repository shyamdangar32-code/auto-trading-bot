# bot/data_io.py
import os
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd

# Zerodha
try:
    from kiteconnect import KiteConnect
    _kite_available = True
except Exception:
    _kite_available = False


# ------------------------- Zerodha helpers -------------------------

def _parse_period_days(period: str) -> int:
    p = period.strip().lower()
    if p.endswith("y"):
        return int(p[:-1]) * 365
    if p.endswith("mo"):
        return int(p[:-2]) * 30
    if p.endswith("d"):
        return int(p[:-1])
    if p in {"6mo", "1y", "5y"}:
        return {"6mo": 180, "1y": 365, "5y": 1825}[p]
    return 365

def _map_interval(interval: str) -> str:
    """Map our intervals to Zerodha intervals."""
    return {
        "1d": "day",
        "1h": "60minute",
        "30m": "30minute",
        "15m": "15minute",
        "5m":  "5minute",
    }.get(interval, "day")

def _get_kite_or_raise() -> KiteConnect:
    if not _kite_available:
        raise RuntimeError("kiteconnect package not available")
    api_key = os.getenv("ZERODHA_API_KEY", "").strip()
    access  = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
    if not api_key or not access:
        raise RuntimeError("ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN missing in env")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    # quick validation
    try:
        _ = kite.profile()
    except Exception as e:
        raise RuntimeError(f"Zerodha token problem: {e}")
    return kite

# ------------------------- Zerodha prices -------------------------

def zerodha_prices(instrument_token: int, period: str, interval: str) -> pd.DataFrame:
    kite = _get_kite_or_raise()

    days = _parse_period_days(period)
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=days)
    tf = _map_interval(interval)

    print(f"🟢 Downloading from Zerodha… token={instrument_token} {tf} {from_dt:%Y-%m-%d}->{to_dt:%Y-%m-%d}")
    candles = kite.historical_data(
        instrument_token=int(instrument_token),
        from_date=from_dt,
        to_date=to_dt,
        interval=tf,
    )
    if not candles:
        raise RuntimeError("Zerodha returned no candles")

    d = pd.DataFrame(candles)
    d.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"
    }, inplace=True)
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.dropna(subset=["Close"])
    print(f"✅ Zerodha OK: {len(d)} rows")
    return d


# ------------------------- Public entry point (Zerodha only) -------------------------

def prices(
    symbol: str,
    period: str,
    interval: str,
    zerodha_enabled: bool = False,
    zerodha_instrument_token: Optional[int] = None,
) -> pd.DataFrame:
    """
    Zerodha-only fetch. We require: zerodha_enabled=True and a valid instrument token.
    """
    if not zerodha_enabled:
        raise RuntimeError("Zerodha is required (zerodha_enabled=false)")
    if not zerodha_instrument_token:
        raise RuntimeError("Missing zerodha_instrument_token in config.yaml")

    return zerodha_prices(int(zerodha_instrument_token), period, interval)
