# bot/data_io.py
from __future__ import annotations

import os
import time
import math
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, date

import pandas as pd

# Zerodha KiteConnect (requirements.txt માં હોવું જોઈએ)
try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None  # GitHub Actions માં missing હોય તો clear error ઉઠાડી દઈએ


# ---- Helpers -----------------------------------------------------------------

_INSTRUMENT_TOKEN_MAP: Dict[str, int] = {
    # Common indices
    "NIFTY": 256265,       # NIFTY 50 index
    "BANKNIFTY": 260105,   # BANKNIFTY index
    # તમે માટે જરૂર હોય તો અહીં વધારાના symbols ઉમેરો
}

_INTERVAL_MAP: Dict[str, str] = {
    "1m": "minute",
    "1min": "minute",
    "minute": "minute",
    "3m": "3minute",
    "5m": "5minute",
    "10m": "10minute",
    "15m": "15minute",
    "30m": "30minute",
    "60m": "60minute",
    "day": "day",
    "1d": "day",
    "daily": "day",
}


def _parse_date(s: str) -> datetime:
    """Accepts 'YYYY-MM-DD' or full ISO. Returns naive UTC datetime."""
    if isinstance(s, datetime):
        return s
    if isinstance(s, date):
        return datetime(s.year, s.month, s.day)
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # last resort: YYYY-MM-DD
        return datetime.strptime(s, "%Y-%m-%d")


def _init_kite() -> KiteConnect:
    if KiteConnect is None:
        raise RuntimeError("kiteconnect library not available. Add 'kiteconnect' to requirements.txt")

    api_key = os.environ.get("ZERODHA_API_KEY", "").strip()
    access_token = os.environ.get("ZERODHA_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        raise RuntimeError("Missing ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN in environment.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def _chunk_ranges(start: datetime, end: datetime, days_per_chunk: int) -> List[Tuple[datetime, datetime]]:
    """
    Zerodha rate-limits historical endpoint. Break the full range into chunks.
    """
    out: List[Tuple[datetime, datetime]] = []
    cur = start
    delta = timedelta(days=days_per_chunk)
    while cur < end:
        nxt = min(cur + delta, end)
        out.append((cur, nxt))
        cur = nxt
    return out


# ---- Public API ---------------------------------------------------------------

def get_zerodha_ohlc(
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    interval: str = "1m",
    tz: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """
    Fetch OHLC from Zerodha historical endpoint and return a pandas DataFrame
    indexed by tz-aware timestamps with columns: Open, High, Low, Close, Volume.

    Args:
        symbol: e.g. "NIFTY", "BANKNIFTY" (mapped to instrument tokens here)
        start, end: "YYYY-MM-DD" or datetime
        interval: e.g. "1m", "5m", "15m", "day"
        tz: timezone for index (default Asia/Kolkata)
    """
    itoken = _INSTRUMENT_TOKEN_MAP.get(symbol.upper())
    if not itoken:
        raise RuntimeError(f"Unknown symbol '{symbol}'. Please add instrument token in data_io._INSTRUMENT_TOKEN_MAP")

    intrv = _INTERVAL_MAP.get(interval, interval)

    sdt = _parse_date(start)
    edt = _parse_date(end)

    kite = _init_kite()

    # Kite historical limits:
    #  - minute data typically limited per call (around 30–45 days). Use 30 to be safe.
    #  - daily can be much larger; but keep a sane chunk too.
    chunk_days = 30 if "minute" in intrv else 365
    ranges = _chunk_ranges(sdt, edt, chunk_days)

    frames: List[pd.DataFrame] = []
    for i, (a, b) in enumerate(ranges, start=1):
        # Kite expects inclusive ranges; keep slight +1min padding on end
        try:
            data = kite.historical_data(
                instrument_token=itoken,
                from_date=a,
                to_date=b,
                interval=intrv,
                continuous=False,
                oi=False,
            )
        except Exception as e:
            # rate-limit/backoff: log & continue
            print(f"⚠️  Skip {a.date()}→{b.date()}: {e}")
            # tiny backoff to be gentle
            time.sleep(0.6)
            continue

        if not data:
            continue

        df = pd.DataFrame(data)
        # standardize column names
        rename = {
            "date": "DateTime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=rename)
        # convert timezone
        if "DateTime" in df.columns:
            dt = pd.to_datetime(df["DateTime"], utc=True)
            try:
                # convert to given timezone while keeping unique index
                dt = dt.dt.tz_convert(tz)
            except Exception:
                # if naive → localize then convert
                dt = dt.dt.tz_localize("UTC").dt.tz_convert(tz)
            df.index = dt
            df = df.drop(columns=["DateTime"])

        # ensure core columns exist
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                df[c] = pd.NA

        frames.append(df[["Open", "High", "Low", "Close", "Volume"]])

        # small sleep to reduce "Too many requests"
        time.sleep(0.25)

    if not frames:
        raise RuntimeError(f"No OHLC data returned for {symbol} {sdt.date()}→{edt.date()} ({intrv}).")

    out = pd.concat(frames).sort_index()

    # Deduplicate if API sent overlapping bars
    out = out[~out.index.duplicated(keep="last")]

    return out
