# bot/data_io.py  (with robust retries + backoff for Zerodha historicals)
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

# --- Retry config (can be tuned via env in Actions secrets) ---
_MAX_RETRIES = int(os.getenv("ZERODHA_MAX_RETRIES", "4"))     # total attempts per chunk
_BACKOFF_SEC = float(os.getenv("ZERODHA_BACKOFF_SEC", "2.0")) # base backoff seconds
_SLEEP_BETWEEN_CHUNKS = float(os.getenv("ZERODHA_CHUNK_SLEEP", "0.20"))  # polite delay


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


def _historical_with_retry(
    kite: "KiteConnect",
    token: int,
    from_ts: pd.Timestamp,
    to_ts: pd.Timestamp,
    interval: str,
    attempt_label: str = "",
) -> list[dict]:
    """
    Call kite.historical_data with retries + exponential backoff.
    Raises the last exception if all attempts fail.
    """
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return kite.historical_data(
                instrument_token=token,
                from_date=from_ts.to_pydatetime(),
                to_date=to_ts.to_pydatetime(),
                interval=interval,
                continuous=False,
                oi=False,
            )
        except Exception as e:  # network/timeouts, throttling, etc.
            last_err = e
            # exponential backoff: 2, 4, 8, ...
            wait = _BACKOFF_SEC * (2 ** (attempt - 1))
            print(
                f"⚠️ Zerodha historical_data error (attempt {attempt}/{_MAX_RETRIES}) "
                f"{attempt_label}: {e}. Retrying in {wait:.1f}s…"
            )
            time.sleep(wait)
    # All attempts failed
    raise RuntimeError(
        f"Failed fetching OHLC after {_MAX_RETRIES} retries {attempt_label}: {last_err}"
    ) from last_err


def get_zerodha_ohlc(symbol: str, start: str, end: str, interval: str = "day") -> pd.DataFrame:
    """
    Fetch OHLC from Zerodha Historical API.

    - Handles Zerodha's minute-data limit (max 60 days per request) by
      chunking the requested period automatically.
    - Retries each chunk with exponential backoff to mitigate transient timeouts.
    - Returns a DataFrame indexed by Date with columns:
      ['Open', 'High', 'Low', 'Close', 'Volume'].
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

    chunks: list[pd.DataFrame] = []
    cur = st

    while cur < en:
        nxt = min(cur + pd.Timedelta(days=per_request_days), en)
        label = f"[{sym} {itv} {cur.date()}→{nxt.date()}]"
        try:
            data = _historical_with_retry(kite, token, cur, nxt, itv, attempt_label=label)
        except Exception as e:
            # surface a clear message with the attempted dates (keeps prior behaviour)
            raise RuntimeError(
                f"Failed fetching OHLC for {sym} ({itv}) {cur.date()} → {nxt.date()}: {e}"
            ) from e

        if data:
            chunks.append(pd.DataFrame(data))

        cur = nxt
        if cur < en:
            time.sleep(__SLEEP_BETWEEN_CHUNKS)

    if not chunks:
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
