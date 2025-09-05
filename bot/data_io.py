# bot/data_io.py
from __future__ import annotations

import os
import pandas as pd

# NOTE: kiteconnect is already in requirements for this repo. If not, add it there.
from kiteconnect import KiteConnect


# --- Helpers -----------------------------------------------------------------

_INSTRUMENT_TOKENS = {
    # Index tokens on Zerodha
    "NIFTY": 256265,       # NIFTY 50 index
    "NIFTY 50": 256265,
    "NSE:NIFTY 50": 256265,
    "BANKNIFTY": 260105,   # NIFTY BANK index
    "NIFTY BANK": 260105,
    "NSE:NIFTY BANK": 260105,
}


def _normalize_symbol(symbol: str) -> str:
    if not symbol:
        return "NIFTY"
    s = symbol.upper().replace("_", " ").strip()
    # common aliases
    if s in ("NIFTY50", "NIFTY-50"):
        return "NIFTY 50"
    if s in ("BANKNIFTY", "NIFTYBANK", "NIFTY-BANK"):
        return "NIFTY BANK"
    return s


def _map_interval(interval: str) -> str:
    """
    Map friendly intervals to Kite historical API values.
    Accepts: '1m','3m','5m','10m','15m','30m','60m','day','week','month'
    """
    if not interval:
        return "day"
    itv = interval.lower().strip()
    mapping = {
        "1m": "minute",
        "3m": "3minute",
        "5m": "5minute",
        "10m": "10minute",
        "15m": "15minute",
        "30m": "30minute",
        "60m": "60minute",
        "1h": "60minute",
        "day": "day",
        "d": "day",
        "week": "week",
        "w": "week",
        "month": "month",
        "mo": "month",
        "m": "minute",   # fallback if someone passes just 'm'
    }
    return mapping.get(itv, itv)


# --- Public API ---------------------------------------------------------------

def get_zerodha_ohlc(symbol: str, start: str, end: str, interval: str = "day") -> pd.DataFrame:
    """
    Fetch OHLC from Zerodha historical API for index symbols (NIFTY/BANKNIFTY).
    Credentials are read from environment:
        ZERODHA_API_KEY
        ZERODHA_ACCESS_TOKEN
    (SECRET is not required for historical endpoint once access token is set)

    Parameters
    ----------
    symbol : str      e.g. 'NIFTY', 'BANKNIFTY'
    start  : 'YYYY-MM-DD'
    end    : 'YYYY-MM-DD'
    interval : str    e.g. '1m', '5m', '15m', 'day', ...

    Returns
    -------
    pd.DataFrame indexed by datetime with columns:
      Open, High, Low, Close, Volume
    """
    sym = _normalize_symbol(symbol)
    token = _INSTRUMENT_TOKENS.get(sym)
    if token is None:
        raise ValueError(f"Unknown/unsupported symbol for Zerodha historicals: {symbol}")

    kite = KiteConnect(api_key=os.environ["ZERODHA_API_KEY"])
    # Access token must be generated outside and provided via secrets
    kite.set_access_token(os.environ["ZERODHA_ACCESS_TOKEN"])

    itv = _map_interval(interval)

    # Convert dates to pandas Timestamps (Kite accepts naive datetimes)
    st = pd.to_datetime(start)
    en = pd.to_datetime(end)

    data = kite.historical_data(
        instrument_token=token,
        from_date=st.to_pydatetime(),
        to_date=en.to_pydatetime(),
        interval=itv,
        continuous=False,
        oi=False,
    )

    if not data:
        # Return empty well-formed frame to let the caller handle "No metrics found"
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(data)
    # Zerodha returns keys: date, open, high, low, close, volume
    df.rename(
        columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"},
        inplace=True,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    return df
