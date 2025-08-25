# runner_intraday_options.py
# Paper intraday options preview using the SAME validated Zerodha client
# used in bot/data_io._get_kite().

from __future__ import annotations

import os
import sys
import json
import math
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# --------- use the exact same validated client as diagnostics ----------
# This reads ZERODHA_API_KEY + ZERODHA_ACCESS_TOKEN from env and calls profile()
from bot.data_io import _get_kite as _kite_ok


def get_kite():
    """Return a ready-to-use KiteConnect client (validated)."""
    kite = _kite_ok()  # prints "✅ Zerodha token OK." if fine, raises otherwise
    return kite


# ----------------------------- helpers --------------------------------
IST = "Asia/Kolkata"


def _today_ist() -> date:
    # runner executes in UTC; we just use 'today' for simplicity
    # (for precise IST, use pytz/zoneinfo if you like)
    return datetime.utcnow().date()


def _next_weekday(d: date, weekday: int) -> date:
    """
    Return the next date with the given weekday (Mon=0..Sun=6).
    If d itself is that weekday, return d.
    """
    days_ahead = (weekday - d.weekday()) % 7
    return d + timedelta(days=days_ahead)


def banknifty_weekly_expiry(base: date | None = None) -> date:
    """
    BANKNIFTY weekly expiry is currently WEDNESDAY.
    (If this changes later, update the weekday below.)
    """
    base = base or _today_ist()
    return _next_weekday(base, 2)  # 2 = Wednesday


def round_to_nearest_strike(spot: float, step: int) -> int:
    return int(round(spot / step) * step)


def instruments_df(kite) -> pd.DataFrame:
    ins = kite.instruments("NFO")
    df = pd.DataFrame(ins)
    # keep common fields
    keep = [
        "instrument_token",
        "tradingsymbol",
        "name",
        "segment",
        "exchange",
        "instrument_type",
        "strike",
        "expiry",
        "lot_size",
        "tick_size",
    ]
    df = df[keep]
    if not pd.api.types.is_datetime64_any_dtype(df["expiry"]):
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df


def find_option_tokens_for_banknifty(
    kite, expiry: date, atm: int
) -> Tuple[int, int, pd.DataFrame]:
    """
    Return (ce_token, pe_token, nfo_df_filtered)
    """
    df = instruments_df(kite)
    f = df[
        (df["name"] == "BANKNIFTY")
        & (df["segment"] == "NFO-OPT")
        & (df["expiry"] == expiry)
        & (df["strike"].astype(int).isin([atm]))
    ]
    ce_row = f[(f["instrument_type"] == "CE")].head(1)
    pe_row = f[(f["instrument_type"] == "PE")].head(1)

    if ce_row.empty or pe_row.empty:
        raise RuntimeError(
            f"Could not find CE/PE tokens for BANKNIFTY {expiry} @ {atm}"
        )

    ce_token = int(ce_row["instrument_token"].iloc[0])
    pe_token = int(pe_row["instrument_token"].iloc[0])
    return ce_token, pe_token, f


def banknifty_spot(kite) -> float:
    # index symbol for LTP in KiteConnect:
    # "NSE:NIFTY BANK" (a.k.a. BANKNIFTY)
    quote = kite.ltp(["NSE:NIFTY BANK"])
    return float(quote["NSE:NIFTY BANK"]["last_price"])


def hist_ohlc(kite, instrument_token: int, frm: datetime, to: datetime, interval="5minute") -> pd.DataFrame:
    candles = kite.historical_data(
        instrument_token=instrument_token,
        from_date=frm,
        to_date=to,
        interval=interval,
    )
    if not candles:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(candles)
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
    return df


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ------------------------------ main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./reports")
    ap.add_argument("--interval", default="5minute")
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--underlying", default="BANKNIFTY")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    kite = get_kite()

    # 1) Spot & expiry
    spot = banknifty_spot(kite)
    expiry = banknifty_weekly_expiry()
    step = 100  # BANKNIFTY strike step
    atm = round_to_nearest_strike(spot, step)

    print(f"ℹ️  {args.underlying} spot {spot:.1f} → ATM {atm}")
    print(f"ℹ️  Weekly expiry → {expiry}")

    # 2) Find ATM CE/PE tokens
    ce_token, pe_token, df_filtered = find_option_tokens_for_banknifty(kite, expiry, atm)
    print(f"ℹ️  CE BANKNIFTY token={ce_token}")
    print(f"ℹ️  PE BANKNIFTY token={pe_token}")

    # 3) Pull today 5-min bars (09:15–15:30 IST roughly → take UTC day span)
    now = datetime.utcnow()
    start = now - timedelta(days=1)  # safe window
    ce_df = hist_ohlc(kite, ce_token, start, now, args.interval)
    pe_df = hist_ohlc(kite, pe_token, start, now, args.interval)
    print(f"ℹ️  Zerodha CE rows: {len(ce_df)} | PE rows: {len(pe_df)}")

    # 4) Tiny indicators (MA20 on Close) just for preview
    for d in (ce_df, pe_df):
        if not d.empty:
            d["MA20"] = d["Close"].rolling(20).mean()

    def last_row_txt(label, d: pd.DataFrame) -> str:
        if d.empty:
            return f"{label} | nan"
        last = d.iloc[-1]
        ma = float(last.get("MA20", float("nan")))
        direction = "UP" if last["Close"] >= ma else "DOWN"
        return f"{label} | {last['Date']} | Close: {last['Close']:.2f} | MA20: {ma:.2f} | {direction}"

    print("—" * 20)
    print("📈", last_row_txt("BANKNIFTY CE", ce_df))
    print("📉", last_row_txt("BANKNIFTY PE", pe_df))
    print("📝 Paper bias (demo):", "PE_BIAS" if not pe_df.empty else "UNKNOWN")
    print("ℹ️  No trades placed. This runner is for data sanity + signals preview.")

    # 5) Write reports
    # latest.json
    summary = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "underlying": args.underlying,
        "spot": spot,
        "atm": atm,
        "expiry": str(expiry),
        "interval": args.interval,
        "ce_token": ce_token,
        "pe_token": pe_token,
        "rows": {"ce": int(len(ce_df)), "pe": int(len(pe_df))},
    }
    (out_dir / "latest.json").write_text(json.dumps(summary, indent=2))

    # latest_signals.csv (very small preview)
    rows = []
    if not ce_df.empty:
        last = ce_df.iloc[-1]
        rows.append(
            {
                "symbol": f"BANKNIFTY{expiry:%d%b%y}".upper() + f"{atm}CE",
                "close": last["Close"],
                "ma20": last.get("MA20", float("nan")),
                "dir": "UP" if last["Close"] >= last.get("MA20", last["Close"]) else "DOWN",
            }
        )
    if not pe_df.empty:
        last = pe_df.iloc[-1]
        rows.append(
            {
                "symbol": f"BANKNIFTY{expiry:%d%b%y}".upper() + f"{atm}PE",
                "close": last["Close"],
                "ma20": last.get("MA20", float("nan")),
                "dir": "UP" if last["Close"] >= last.get("MA20", last["Close"]) else "DOWN",
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "latest_signals.csv", index=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # make failures obvious in Actions log
        print("❌ Runner failed:", repr(e))
        sys.exit(1)
