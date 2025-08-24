# runner_intradary_options.py
"""
Intraday options (paper) runner — Zerodha only

What this script does (safe for GitHub Actions):
1) Connects to Zerodha using env secrets (API key + ACCESS_TOKEN).
2) Finds BANKNIFTY spot, computes nearest-100 ATM strike.
3) Locates the weekly expiry CE & PE instruments and prints their tokens.
4) Downloads recent 5-minute candles for the CE & PE.
5) FIX: Robust datetime normalization
   - Zerodha returns tz-aware datetimes; we now use tz_convert (NOT tz_localize)
   - Accepts columns named 'date' or 'timestamp' (no more KeyError: 'Date')
6) A tiny demo “paper signal” (no orders): prints last price + a HOLD/EXIT message.
"""

import os
from datetime import datetime, timedelta, date
from typing import Optional, Tuple

import pandas as pd

# Zerodha
from kiteconnect import KiteConnect


TZ = "Asia/Kolkata"
INDEX_NAME = "BANKNIFTY"         # change to NIFTY if you like
INTERVAL = "5minute"
LOOKBACK_DAYS = 5                # how much intraday history to pull

# ------------- helpers ------------- #

def _get_kite() -> KiteConnect:
    api_key = os.getenv("ZERODHA_API_KEY", "").strip()
    access  = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
    if not api_key or not access:
        raise RuntimeError("ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN are missing in env.")
    k = KiteConnect(api_key=api_key)
    k.set_access_token(access)
    # quick token check
    _ = k.profile()
    print("✅ Zerodha token OK.")
    return k

def _next_thursday(d: date) -> date:
    # Zerodha weekly index options typically expire on Thursday
    # Return the coming Thursday (or today if already Thursday)
    wd = d.weekday()  # Mon=0 ... Sun=6
    add = (3 - wd) % 7  # 3 => Thursday
    return d + timedelta(days=add)

def _round_to_100(x: float) -> int:
    return int(round(x / 100.0) * 100)

def _normalize_dt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust datetime column normalize:
    - Accept 'date' or 'timestamp' or 'datetime'
    - Convert to tz-aware Asia/Kolkata using tz_convert if already aware
    - Expose a 'Date' column (string ISO) and keep index as pandas datetime
    """
    d = df.copy()
    # locate the source dt col
    dtcol = None
    for c in ["date", "timestamp", "datetime", "Date", "DATE"]:
        if c in d.columns:
            dtcol = c
            break
    if dtcol is None:
        raise KeyError(f"No datetime column found in columns: {list(d.columns)[:10]}")

    # to pandas datetime
    dt = pd.to_datetime(d[dtcol], errors="coerce", utc=True)

    # If timezone-aware already, convert; else localize first
    # pd.to_datetime(..., utc=True) returns tz-aware UTC
    # so we simply convert to Asia/Kolkata.
    dt = dt.dt.tz_convert(TZ)

    d.index = dt
    d["Date"] = dt.astype(str)  # ISO with timezone
    return d

def _find_option_tokens(kite: KiteConnect, index_name: str, expiry: date, atm: int) -> Tuple[int, int, str, str]:
    """
    Return (ce_token, pe_token, ce_symbol, pe_symbol) for given index, weekly expiry and ATM strike.
    """
    inst = pd.DataFrame(kite.instruments("NFO"))
    inst = inst[(inst["segment"] == "NFO-OPT") & (inst["name"] == index_name)]

    inst["expiry"] = pd.to_datetime(inst["expiry"]).dt.date
    ce = inst[(inst["expiry"] == expiry) & (inst["strike"] == atm) & (inst["instrument_type"] == "CE")].copy()
    pe = inst[(inst["expiry"] == expiry) & (inst["strike"] == atm) & (inst["instrument_type"] == "PE")].copy()
    if ce.empty or pe.empty:
        raise RuntimeError(f"Could not find CE/PE for {index_name} {atm}@{expiry}.")
    ce = ce.iloc[0]
    pe = pe.iloc[0]
    return ce["instrument_token"], pe["instrument_token"], ce["tradingsymbol"], pe["tradingsymbol"]

def _hist(kite: KiteConnect, token: int, days: int, interval: str) -> pd.DataFrame:
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(days=days)
    candles = kite.historical_data(instrument_token=token, from_date=from_dt, to_date=to_dt, interval=interval)
    if not candles:
        raise RuntimeError("No candles returned from Zerodha.")
    d = pd.DataFrame(candles)
    # standardize OHLCV names
    d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                      "close": "Close", "volume": "Volume"}, inplace=True)
    d = _normalize_dt(d)
    d = d.dropna(subset=["Close"])
    return d

def _banknifty_spot(kite: KiteConnect) -> float:
    # Zerodha quote key for NIFTY BANK index (works in live):
    q = kite.quote("NSE:NIFTY BANK")
    # example structure: {'NSE:NIFTY BANK': {'last_price': ... , ...}}
    item = next(iter(q.values()))
    return float(item["last_price"])

# ------------- main ------------- #

def main():
    kite = _get_kite()

    # 1) Spot & ATM
    spot = _banknifty_spot(kite)
    atm = _round_to_100(spot)
    expiry = _next_thursday(date.today())
    print(f"ℹ️ {INDEX_NAME} spot {spot} → ATM {atm}")
    print(f"ℹ️ Weekly expiry -> {expiry}")

    # 2) Locate tokens
    ce_token, pe_token, ce_sym, pe_sym = _find_option_tokens(kite, INDEX_NAME, expiry, atm)
    print(f"🟦 CE {ce_sym} token={ce_token}")
    print(f"🟥 PE {pe_sym} token={pe_token}")

    # 3) Download recent intraday candles
    ce = _hist(kite, ce_token, LOOKBACK_DAYS, INTERVAL)
    pe = _hist(kite, pe_token, LOOKBACK_DAYS, INTERVAL)
    print(f"✅ Zerodha CE rows: {len(ce)}  | PE rows: {len(pe)}")

    # 4) Simple no-trade “paper signal” demo:
    #    We just compare today's VWAP-ish proxy: rolling price mean on last 20 bars.
    ce["ma20"] = ce["Close"].rolling(20, min_periods=1).mean()
    pe["ma20"] = pe["Close"].rolling(20, min_periods=1).mean()

    ce_last = ce.iloc[-1]
    pe_last = pe.iloc[-1]

    ce_state = "UP" if ce_last["Close"] > ce_last["ma20"] else "DOWN"
    pe_state = "UP" if pe_last["Close"] > pe_last["ma20"] else "DOWN"

    print("—" * 60)
    print(f"📈 {ce_sym} | {ce_last['Date']} | Close: {ce_last['Close']:.2f} | MA20: {ce_last['ma20']:.2f} | {ce_state}")
    print(f"📉 {pe_sym} | {pe_last['Date']} | Close: {pe_last['Close']:.2f} | MA20: {pe_last['ma20']:.2f} | {pe_state}")

    # Very conservative paper “bias” message (no orders placed)
    bias = "HOLD"
    if ce_state == "UP" and pe_state == "DOWN":
        bias = "CE_BIAS"
    elif ce_state == "DOWN" and pe_state == "UP":
        bias = "PE_BIAS"

    print(f"📝 Paper bias (demo): {bias}")
    print("ℹ️ No trades placed. This runner is for data sanity + signals preview.")

if __name__ == "__main__":
    main()
