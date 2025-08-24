# runner_intraday_options.py
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, time, timedelta

import pandas as pd

from kiteconnect import KiteConnect
from bot.config import get_cfg
from bot.utils import ensure_dir, send_telegram, save_json
from bot.strategy import StraddleParams, simulate_short_straddle_intraday

IST = "Asia/Kolkata"

def _kite() -> Optional[KiteConnect]:
    api_key = os.getenv("ZERODHA_API_KEY", "").strip()
    access = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
    if not api_key or not access:
        print("❌ Missing ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN")
        return None
    k = KiteConnect(api_key=api_key)
    k.set_access_token(access)
    try:
        _ = k.profile()
        print("✅ Zerodha token OK.")
        return k
    except Exception as e:
        print("❌ Zerodha token problem:", e)
        return None

def _parse_date(dstr: str, tz: str = IST) -> pd.Timestamp:
    if not dstr:
        # today IST
        now_ist = pd.Timestamp.now(tz=tz)
        return now_ist.normalize()
    return pd.Timestamp(dstr).tz_localize(tz) if pd.Timestamp(dstr).tzinfo is None else pd.Timestamp(dstr).tz_convert(tz)

def _ist_ts(d: pd.Timestamp, hhmm: str) -> pd.Timestamp:
    hh, mm = map(int, hhmm.split(":"))
    ts = d + pd.Timedelta(hours=hh, minutes=mm)
    # ensure tz-aware IST
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)

def _get_underlying_ltp(kite: KiteConnect, underlying: str) -> float:
    # Zerodha index tradingsymbols:
    # NIFTY 50 index token ~ 256265? (but you already use 256265 for NIFTYBEES earlier)
    # Safer: use `kite.quote` with the index name:
    # NSE:NIFTY BANK  OR  NSE:NIFTY 50
    symbol = "NSE:NIFTY BANK" if underlying.upper() == "BANKNIFTY" else "NSE:NIFTY 50"
    q = kite.quote([symbol])
    for v in q.values():
        return float(v["last_price"])
    raise RuntimeError("Unable to get underlying LTP")

def _atm_strike(ltp: float, step: int) -> int:
    return int(round(ltp / step) * step)

def _find_option_tokens(kite: KiteConnect, underlying: str, trade_date: pd.Timestamp,
                        atm: int) -> Tuple[int, int, str]:
    """
    Find CE/PE instrument_tokens for BANKNIFTY/NIFTY weekly of the given trade_date week.
    """
    # expiry: choose the nearest *weekly* expiry on/after trade_date
    nfo = kite.instruments("NFO")
    # Determine lot size & strike step
    if underlying.upper() == "BANKNIFTY":
        prefix = "BANKNIFTY"
        step = 100
    else:
        prefix = "NIFTY"
        step = 50

    # find the weekly expiry >= trade_date
    wd = trade_date.tz_convert(IST).date()
    expiries = sorted({i["expiry"] for i in nfo if i["name"] == prefix})
    expiry = None
    for e in expiries:
        if e >= wd:
            expiry = e
            break
    if expiry is None:
        raise RuntimeError("No weekly expiry found")

    # Build tradingsymbols like BANKNIFTY25AUG55100CE
    # Zerodha uses format: BANKNIFTY<DDMMM><STRIKE><CE/PE>
    # We'll search by name/strike/option_type instead of composing raw string.
    ce_token = pe_token = None
    for i in nfo:
        if i["name"] != prefix:
            continue
        if i["expiry"] != expiry:
            continue
        if i["strike"] == atm and i["instrument_type"] in ("CE", "PE"):
            if i["instrument_type"] == "CE":
                ce_token = i["instrument_token"]
            else:
                pe_token = i["instrument_token"]
        if ce_token and pe_token:
            break

    if not ce_token or not pe_token:
        raise RuntimeError("Could not find ATM CE/PE tokens")

    return ce_token, pe_token, expiry.isoformat()

def _hist(kite: KiteConnect, token: int, frm: pd.Timestamp, to: pd.Timestamp, interval: str) -> pd.DataFrame:
    candles = kite.historical_data(
        instrument_token=token,
        from_date=frm.to_pydatetime(),
        to_date=to.to_pydatetime(),
        interval=interval
    )
    d = pd.DataFrame(candles)
    # Normalize columns
    d.rename(columns={"date": "Date", "open": "Open", "high": "High",
                      "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    d["Date"] = pd.to_datetime(d["Date"], infer_datetime_format=True)
    # IMPORTANT: don't double-localize -> if tz-naive, localize; else convert
    if d["Date"].dt.tz is None:
        d["Date"] = d["Date"].dt.tz_localize(IST)
    else:
        d["Date"] = d["Date"].dt.tz_convert(IST)
    return d

def main():
    cfg = get_cfg()
    OUT = cfg.get("out_dir", "reports")
    ensure_dir(OUT)

    ico = cfg.get("intraday_options", {})
    underlying = str(ico.get("underlying", "BANKNIFTY")).upper()
    lots       = int(ico.get("lots", 1))
    lot_size   = int(ico.get("lot_size", 15 if underlying=="BANKNIFTY" else 50))
    interval   = str(ico.get("interval", "5minute"))
    entry_hhmm = str(ico.get("entry_time", "09:30"))
    exit_hhmm  = str(ico.get("squareoff", "15:10"))
    trade_date = _parse_date(str(ico.get("date", "")), IST)

    leg_sl_pct = float(ico.get("leg_sl_percent", 30.0))
    tgt_pct    = float(ico.get("combined_target_percent", 50.0))

    print("CFG.intraday_options:", {
        "underlying": underlying, "lots": lots, "lot_size": lot_size,
        "interval": interval, "date": str(trade_date.date()),
        "entry": entry_hhmm, "squareoff": exit_hhmm,
        "leg_sl%": leg_sl_pct, "target%": tgt_pct
    })

    kite = _kite()
    if kite is None:
        sys.exit(1)

    # Get underlying LTP and decide ATM
    ltp = _get_underlying_ltp(kite, underlying)
    step = 100 if underlying == "BANKNIFTY" else 50
    atm = _atm_strike(ltp, step)
    print(f"🟦 {underlying} spot {ltp} → ATM {atm}")

    ce_token, pe_token, expiry = _find_option_tokens(kite, underlying, trade_date, atm)
    print(f"CE token={ce_token}  PE token={pe_token}  (expiry {expiry})")

    # Time window for the day
    start_ts = _ist_ts(trade_date, "09:15")
    entry_ts = _ist_ts(trade_date, entry_hhmm)
    end_ts   = _ist_ts(trade_date, exit_hhmm)

    # Download 5m candles
    ce_df = _hist(kite, ce_token, start_ts, end_ts, interval)
    pe_df = _hist(kite, pe_token, start_ts, end_ts, interval)
    print(f"Fetched: CE {len(ce_df)} bars, PE {len(pe_df)} bars.")

    params = StraddleParams(
        lots=lots,
        lot_size=lot_size,
        leg_sl_pct=leg_sl_pct,
        combined_target_pct=tgt_pct,
        entry_ts=entry_ts,
        squareoff_ts=end_ts
    )
    metrics = simulate_short_straddle_intraday(ce_df, pe_df, params)

    print("📈 Intraday options (paper) result:", metrics)

    # Save report
    payload = {
        "timestamp": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
        "underlying": underlying,
        "atm_strike": atm,
        "expiry": expiry,
        "params": params.__dict__,
        "metrics": metrics,
    }
    save_json(payload, os.path.join(OUT, "intraday_options_latest.json"))

    # Optional Telegram ping
    msg = (f"🧪 Intraday {underlying} ATM straddle (paper)\n"
           f"Date: {trade_date.date()}  Exp: {expiry}\n"
           f"Entry {params.entry_ts.time()}  Exit {params.squareoff_ts.time()}\n"
           f"Lots: {params.lots}  LotSize: {params.lot_size}\n"
           f"PnL: ₹{metrics.get('pnl_rs', 0)}  Status: {metrics.get('status')}")
    try:
        send_telegram(msg)
    except Exception as _:
        pass

if __name__ == "__main__":
    # allow --out_dir override (kept for parity)
    if "--out_dir" in sys.argv:
        # bot.config/get_cfg already respects CLI via env in your setup; no-op here
        pass
    main()
