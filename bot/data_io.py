# bot/data_io.py
import time
from typing import Optional
import pandas as pd
import yfinance as yf


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure we have a numeric Close and a Date column, even if yfinance
    returns MultiIndex columns like 'Close|^NSEI'.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["|".join(map(str, c)).strip() for c in df.columns]
    else:
        df = df.copy()
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

    # enforce numeric Close and a Date column
    df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.reset_index()  # yfinance puts DatetimeIndex by default
    df = df.dropna(subset=["Close"])
    return df


def yahoo_prices(symbol: str, period: str, interval: str, retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """
    Robust Yahoo downloader with retries + normalization.
    Example: yahoo_prices("^NSEI", "6mo", "1d")
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

    # If we got here, all retries failed
    raise RuntimeError(str(last_err) if last_err else f"Yahoo failed for {symbol}")
