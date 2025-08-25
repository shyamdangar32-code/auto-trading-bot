# bot/data_io.py
import os
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
from kiteconnect import KiteConnect

# ------------------------- helpers -------------------------

def _parse_period_days(period: str) -> int:
    p = (period or "").strip().lower()
    if p.endswith("y"):  return int(p[:-1]) * 365
    if p.endswith("mo"): return int(p[:-2]) * 30
    if p.endswith("d"):  return int(p[:-1])
    return 365

def _map_interval(interval: str) -> str:
    return {
        "1d": "day",
        "1h": "60minute",
        "30m": "30minute",
        "15m": "15minute",
        "5m": "5minute",
        "5minute": "5minute",
    }.get((interval or "").lower(), "day")

def _get_kite() -> KiteConnect:
    api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()

    # Early sanity checks (so errors are obvious)
    if len(api_key) < 6:
        raise RuntimeError("ZERODHA_API_KEY missing/too short in environment.")
    if len(access) < 10:
        raise RuntimeError("ZERODHA_ACCESS_TOKEN missing/too short in environment.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)

    # Verify session; if this fails it's almost always bad token
    kite.profile()
    print("✅ Zerodha token OK.")
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
    df.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"
    }, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("Asia/Kolkata")
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
    if not zerodha_enabled:
        raise RuntimeError("Zerodha disabled in config; Yahoo fallback removed.")

    if not zerodha_instrument_token:
        raise RuntimeError("Missing `zerodha_instrument_token` in config.yaml")

    return zerodha_prices(int(zerodha_instrument_token), period, interval)
