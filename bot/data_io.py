# bot/data_io.py
from __future__ import annotations

import os
import time
import typing as t
import pandas as pd

try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception as e:  # pragma: no cover
    KiteConnect = None  # allow import in non-runtime contexts


# --- Known index tokens (spot indices on Zerodha) ---
# Reference: NIFTY 50 = 256265, BANKNIFTY = 260105
_INSTRUMENT_TOKENS: dict[str, int] = {
    "NIFTY": 256265,
    "NIFTY50": 256265,
    "BANKNIFTY": 260105,
    "NIFTYBANK": 260105,
}


def _normalize_symbol(s: str) -> str:
    """Normalize common index aliases to Zerodha keys used above."""
    if not s:
        raise ValueError("symbol is empty")
    s = s.strip().upper()
    # Remove spaces / hyphens like "NIFTY 50" → "NIFTY50"
    s = s.replace(" ", "").replace("-", "")
    # map back a couple of forms
    if s in ("NIFTY", "NIFTY50"):
        return "NIFTY"
    if s in ("BANKNIFTY", "NIFTYBANK"):
        return "BANKNIFTY"
    return s


def _map_interval(interval: str) -> str:
    """
    Map user-friendly intervals to Zerodha historical API intervals.
    Accepts: '1m', '3m', '5m', '15m', '30m', '60m', 'day', '1d', 'week', 'month'
             or TA-lib/ccxt-like '1minute', '5minute', etc.
    """
    iv = (interval or "").strip().lower()
    aliases = {
        "1m": "minute",
        "1min": "minute",
        "1minute": "minute",
        "3m": "3minute",
        "3min": "3minute",
        "3minute": "3minute",
        "5m": "5minute",
        "5min": "5minute",
        "5minute": "5minute",
        "10m": "10minute",
        "10min": "10minute",
        "10minute": "10minute",
        "15m": "15minute",
        "15min": "15minute",
        "15minute": "15minute",
        "30m": "30minute",
        "30min": "30minute",
        "30minute": "30minute",
        "60m": "60minute",
        "60min": "60minute",
        "60minute": "60minute",
        "1h": "60minute",
        "day": "day",
        "1d": "day",
        "d": "day",
        "week": "week",
        "w": "week",
        "month": "month",
        "mo": "month",
    }
    return aliases.get(iv, iv or "day")


def _require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(
            f"Missing environment variable: {name}. "
            "Set repo → Settings → Secrets and variables → Actions."
        )
    return v


def _build_kite() -> "KiteConnect":
    if KiteConnect is None:
        raise RuntimeError(
            "kiteconnect is not installed. Ensure requirements.txt includes 'kiteconnect'."
        )
    api_key = _require_env("ZERODHA_API_KEY")
    access_token = _require_env("ZERODHA_ACCESS_TOKEN")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def get_zerodha_ohlc(symbol: str, start: str, end: str, interval: str = "day") -> pd.DataFrame:
    """
    Fetch OHLC from Zerodha Historical API.

    - Handles Zerodha's minute-data limit (max 60 days per request) by
      chunking the requested period automatically.
    - Returns a DataFrame indexed by Date with columns:
      ['Open', 'High', 'Low', 'Close', 'Volume'].

    Parameters
    ----------
    symbol : str
        e.g. 'NIFTY' or 'BANKNIFTY'
    start, end : 'YYYY-MM-DD' (or any pandas-parsable date)
    interval : str
        '1m','3m','5m','15m','30m','60m','day','week','month' supported.
    """
    sym = _normalize_symbol(symbol)
    token = _INSTRUMENT_TOKENS.get(sym)
    if token is None:
        raise ValueError(f"Unknown/unsupported symbol for Zerodha historicals: {symbol}")

    kite = _build_kite()

    itv = _map_interval(interval)
    st = pd.to_datetime(start)
    en = pd.to_datetime(end)
    if st >= en:
        raise ValueError(f"start ({start}) must be < end ({end})")

    # Zerodha minute endpoints limit: 60 days per request
    per_request_days = 60 if "minute" in itv else 5000  # practically no limit for day/week/month
    sleep_between = 0.20  # small delay to be nice to the API

    chunks: list[pd.DataFrame] = []
    cur = st

    while cur < en:
        nxt = min(cur + pd.Timedelta(days=per_request_days), en)
        # Perform one chunked request
        try:
            data = kite.historical_data(
                instrument_token=token,
                from_date=cur.to_pydatetime(),
                to_date=nxt.to_pydatetime(),
                interval=itv,
                continuous=False,
                oi=False,
            )
        except Exception as e:
            # surface a clear message with the attempted dates
            raise RuntimeError(
                f"Failed fetching OHLC for {sym} ({itv}) "
                f"{cur.date()} → {nxt.date()}: {e}"
            ) from e

        if data:
            chunks.append(pd.DataFrame(data))
        cur = nxt
        if cur < en:
            time.sleep(sleep_between)

    if not chunks:
        # Return empty-but-shaped DataFrame
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        empty.index.name = "Date"
        return empty

    df = pd.concat(chunks, ignore_index=True)

    # Normalize columns to a standard OHLCV shape
    df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    # Some instruments may return tz-aware stamps; drop tz for consistent downstream ops
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df
