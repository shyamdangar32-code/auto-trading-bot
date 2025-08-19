import pandas as pd
import yfinance as yf

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["|".join(map(str, c)) for c in df.columns]
    df = df.reset_index()
    # Map expected OHLCV
    for k in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if k not in df.columns:
            cand = [c for c in df.columns if c.split("|")[0].lower()==k.lower()]
            if cand:
                df[k] = df[cand[0]]
    for k in ["Open","High","Low","Close","Adj Close","Volume"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    df = df.dropna(subset=["Close"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def yahoo_prices(symbol: str, period: str, interval: str) -> pd.DataFrame:
    print("🟡 Downloading from Yahoo Finance…")
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} ({period},{interval})")
    df = _normalize(df)
    print("✅ Rows:", len(df))
    return df
