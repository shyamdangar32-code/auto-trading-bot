# runner_intraday_options.py
# Intraday options (paper) – robust expiry + token finding

import os
from datetime import datetime, date, timedelta, time as dt_time
from typing import Tuple, Optional

import pandas as pd
from kiteconnect import KiteConnect

# ---------- Config (reads from config.yaml via env-injection in workflow) ----------
OUT_DIR = os.getenv("OUT_DIR", "reports")
UNDERLYING = os.getenv("INTRA_UNDERLYING", "BANKNIFTY").upper()   # "BANKNIFTY" | "NIFTY"
INTERVAL   = os.getenv("INTRA_INTERVAL",  "5minute")

# strike step & lot size hints (just for logs)
STRIKE_STEP = 100 if UNDERLYING == "BANKNIFTY" else 50

# ---------- Zerodha helpers ----------
def get_kite() -> KiteConnect:
    api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise RuntimeError("Missing ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN in env.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    # Sanity check
    kite.profile()
    print("✅ Zerodha token OK.")
    return kite

def next_thursday(d: date) -> date:
    # Monday=0 ... Thursday=3 ... Sunday=6
    target_wd = 3
    days = (target_wd - d.weekday()) % 7
    if days == 0:
        # If it's already Thursday, keep the same day (weekly expiry is today)
        return d
    return d + timedelta(days=days)

def round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)

def option_tokens_for_atm(
    inst_df: pd.DataFrame, underlying: str, expiry: date, atm: int
) -> Tuple[Optional[int], Optional[int]]:
    """
    Try to find CE & PE instrument_token for a given UNDERLYING / expiry / ATM strike.
    If not found on exact expiry, tries expiry±1 day.
    Returns (ce_token, pe_token) or (None, None).
    """

    def _filter_for_exp(dt_exp: date) -> pd.DataFrame:
        f = inst_df[
            (inst_df["segment"] == "NFO-OPT")
            & (inst_df["name"].str.upper() == underlying)
            & (pd.to_datetime(inst_df["expiry"]).dt.date == dt_exp)
            & (inst_df["strike"] == float(atm))
        ]
        return f

    # 1) Exact expiry
    e0 = _filter_for_exp(expiry)
    print(f"🔎 Instruments @ {expiry}: {len(e0)} rows for strike {atm}")

    # 2) Fallbacks: expiry - 1 day, expiry + 1 day
    e_minus = _filter_for_exp(expiry - timedelta(days=1)) if e0.empty else pd.DataFrame()
    if e0.empty:
        print(f"   …fallback {expiry - timedelta(days=1)}: {len(e_minus)} rows")
    e_plus  = _filter_for_exp(expiry + timedelta(days=1)) if e0.empty and e_minus.empty else pd.DataFrame()
    if e0.empty and e_minus.empty:
        print(f"   …fallback {expiry + timedelta(days=1)}: {len(e_plus)} rows")

    pool = e0
    chosen_exp = expiry
    if pool.empty and not e_minus.empty:
        pool = e_minus
        chosen_exp = expiry - timedelta(days=1)
    if pool.empty and not e_plus.empty:
        pool = e_plus
        chosen_exp = expiry + timedelta(days=1)

    if pool.empty:
        return None, None

    ce = pool.loc[pool["instrument_type"] == "CE", "instrument_token"]
    pe = pool.loc[pool["instrument_type"] == "PE", "instrument_token"]

    ce_token = int(ce.iloc[0]) if not ce.empty else None
    pe_token = int(pe.iloc[0]) if not pe.empty else None

    print(f"✅ Using expiry {chosen_exp} | CE token: {ce_token} | PE token: {pe_token}")
    return ce_token, pe_token

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    kite = get_kite()

    # 1) Get spot to compute ATM
    # BANKNIFTY index token for LTP is available via quote for 'NSE:NIFTY BANK' alias in some libs,
    # but using instrument token is simpler through the instruments file.
    # We’ll use LTP via kite.ltp for index tradingsymbols supported by Kite.
    # For BANKNIFTY/NIFTY, Kite supports "NSE:NIFTY BANK" and "NSE:NIFTY 50".
    spot_tradingsymbol = "NSE:NIFTY BANK" if UNDERLYING == "BANKNIFTY" else "NSE:NIFTY 50"
    spot_ltp = kite.ltp([spot_tradingsymbol])[spot_tradingsymbol]["last_price"]
    atm = round_to_step(spot_ltp, STRIKE_STEP)

    # 2) Compute intended weekly expiry (Thursday)
    today = datetime.now().date()
    exp = next_thursday(today)

    print(f"ℹ️  {UNDERLYING} spot {spot_ltp:.1f} → ATM {atm}")
    print(f"ℹ️  Weekly expiry → {exp}")

    # 3) Pull NFO instruments and hunt for tokens
    inst = pd.DataFrame(kite.instruments("NFO"))
    # Keep only needed columns to speed up ops
    keep = ["instrument_token", "tradingsymbol", "name", "segment", "expiry", "strike", "instrument_type"]
    inst = inst[keep].copy()
    inst["name"] = inst["name"].astype(str).str.upper()

    ce_token, pe_token = option_tokens_for_atm(inst, UNDERLYING, exp, atm)
    if not ce_token or not pe_token:
        raise RuntimeError(f"Could not find CE/PE tokens for {UNDERLYING} {exp} @ {atm}")

    # 4) Pull a small OHLC preview to emulate your previous “signals preview” output
    def hist_df(token: int) -> pd.DataFrame:
        # 3 days of 5-min candles is enough for a preview here
        frm = datetime.now() - timedelta(days=3)
        candles = kite.historical_data(
            instrument_token=token,
            from_date=frm,
            to_date=datetime.now(),
            interval="5minute",
        )
        df = pd.DataFrame(candles)
        if df.empty:
            return df
        df.rename(columns={"date": "Date", "open": "Open", "high": "High",
                           "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    ce_df = hist_df(ce_token)
    pe_df = hist_df(pe_token)
    print("—" * 42)
    print(f"📈 {UNDERLYING}{exp:%d%b}{atm:05d}CE | rows: {len(ce_df)} | Close: {ce_df['Close'].iloc[-1] if not ce_df.empty else 'nan'}")
    print(f"📉 {UNDERLYING}{exp:%d%b}{atm:05d}PE | rows: {len(pe_df)} | Close: {pe_df['Close'].iloc[-1] if not pe_df.empty else 'nan'}")

    # 5) Minimal paper-run banner (no live orders here)
    print("🧪 Paper bias (demo): PE_BIAS" if (not pe_df.empty and not ce_df.empty and pe_df['Close'].iloc[-1] > ce_df['Close'].iloc[-1]) else "🧪 Paper bias (demo): CE_BIAS")
    print("ℹ️  No trades placed. This runner is for data sanity + signals preview.")

    # 6) Write a tiny report so the artifact always exists
    rpt = {
        "underlying": UNDERLYING,
        "spot": spot_ltp,
        "atm": atm,
        "expiry": exp.isoformat(),
        "ce_token": ce_token,
        "pe_token": pe_token,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    pd.Series(rpt).to_json(os.path.join(OUT_DIR, "latest.json"), indent=2)
    print(f"🗂  Wrote {os.path.join(OUT_DIR, 'latest.json')}")

if __name__ == "__main__":
    main()
