# bot/data_io.py
import os
import time
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Zerodha (optional)
try:
    from kiteconnect import KiteConnect
    _kite_available = True
except Exception:
    _kite_available = False


# ------------------------- helpers -------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we end up with numeric Close and a Date column even if yfinance
    returns MultiIndex columns like 'Close|^NSEI'.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["|".join(map(str, c)).strip() for c in df.columns]
    else:
        df.columns = [str(c).strip() for c in df.columns]

    # Find any variant of Close
    close_col: Optional[str] = None

    # 1) exact 'Close'
    for c in df.columns:
        if c.lower() == "close":
            close_col = c
            break
    # 2) prefix 'Close|...'
    if close_col is None:
        for c in df.columns:
            if c.lower().startswith("close|"):
                close_col = c
                break
    # 3) safety: split by '|' and check first token
    if close_col is None:
        for c in df.columns:
            parts = c.split("|")
            if parts and parts[0].lower() == "close":
                close_col = c
                break

    if close_col is None:
        raise RuntimeError(f"'Close' column missing. Columns: {list(df.columns)[:10]} ...")

    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    if "Date" not in df.columns:
        df = df.reset_index()  # yfinance usually gives DatetimeIndex
    df = df.dropna(subset=["Close"])
    return df


# ------------------------- Yahoo -------------------------

def yahoo_prices(symbol: str, period: str, interval: str, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """
    Robust Yahoo downloader with retries + normalization.
    Example: yahoo_prices("^NSEI", "1y", "1d")
    """
    last_err = None
    print(f"🟡 Downloading from Yahoo Finance… symbol={symbol} period={period} interval={interval}")
    for i in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
            if df is None or df.empty:
                raise RuntimeError(f"No data for {symbol} ({period},{interval})")

            df = _normalize_columns(df)
            print(f"✅ Yahoo OK: {len(df)} rows")
            return df

        except Exception as e:
            last_err = e
            if i < retries:
                print(f"⚠️ Yahoo attempt {i} failed: {e}\n⏳ Retrying in {delay}s…")
                time.sleep(delay)
            else:
                print("🛑 Yahoo failed after retries.")
    raise RuntimeError(str(last_err) if last_err else f"Yahoo failed for {symbol}")


# ------------------------- Zerodha -------------------------

def _parse_period_days(period: str) -> int:
    period = period.strip().lower()
    if period.endswith("y"):
        return int(period[:-1]) * 365
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    if period.endswith("d"):
        return int(period[:-1])
    # defaults
    if period in {"6mo", "1y", "5y"}:
        return {"6mo": 180, "1y": 365, "5y": 1825}[period]
    return 365

def _map_interval(interval: str) -> str:
    """Map Yahoo-style intervals to Zerodha intervals."""
    m = {
        "1d": "day",
        "1h": "60minute",
        "30m": "30minute",
        "15m": "15minute",
        "5m": "5minute",
    }
    return m.get(interval, "day")

def _get_kite():
    if not _kite_available:
        print("ℹ️ kiteconnect not available; skipping Zerodha.")
        return None
    api_key = os.getenv("ZERODHA_API_KEY", "").strip()
    access  = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
    if not api_key or not access:
        print("ℹ️ ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN not set; skipping Zerodha.")
        return None
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    try:
        _ = kite.profile()
        print("✅ Zerodha token OK.")
        return kite
    except Exception as e:
        print("❌ Zerodha token problem:", e)
        return None

def zerodha_prices(instrument_token: int, period: str, interval: str) -> pd.DataFrame:
    kite = _get_kite()
    if kite is None:
        raise RuntimeError("Zerodha unavailable")

    days = _parse_period_days(period)
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=days)
    tf = _map_interval(interval)

    print(f"🟢 Downloading from Zerodha… token={instrument_token} {tf} {from_dt:%Y-%m-%d}->{to_dt:%Y-%m-%d}")
    candles = kite.historical_data(
        instrument_token=instrument_token,
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


# ------------------------- Orchestrator -------------------------

def prices(symbol: str,
           period: str,
           interval: str,
           zerodha_enabled: bool = False,
           zerodha_instrument_token: Optional[int] = None) -> pd.DataFrame:
    """
    Unified entry point:
      - If zerodha_enabled and instrument token present: try Zerodha first,
        fallback to Yahoo.
      - Otherwise: Yahoo only (old behavior).
    """
    if zerodha_enabled and zerodha_instrument_token:
        try:
            return zerodha_prices(int(zerodha_instrument_token), period, interval)
        except Exception as ze:
            print(f"🟠 Zerodha failed: {ze} — falling back to Yahoo.")
            # fall through to Yahoo
    # Yahoo path
    return yahoo_prices(symbol, period, interval)
