#!/usr/bin/env python3
# runner_intraday_options.py

import os
import json
import math
import argparse
from types import SimpleNamespace
from datetime import datetime, date, time, timedelta, timezone

import pandas as pd
from kiteconnect import KiteConnect

# ---------------- utils ----------------

IST = timezone(timedelta(hours=5, minutes=30))

def ensure_date(x):
    return x.date() if isinstance(x, datetime) else x

def ist_now():
    return datetime.now(IST)

def is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Sat=5, Sun=6

def prev_trading_day(d: date) -> date:
    dd = d - timedelta(days=1)
    while is_weekend(dd):
        dd -= timedelta(days=1)
    return dd

def round_to_strike(spot: float, base: int) -> int:
    return int(round(spot / base) * base)

def parse_hhmm(s: str, fallback: str = "09:30") -> tuple[int,int]:
    """Return (hh,mm); if missing/invalid, use fallback."""
    txt = (s or "").strip()
    if not txt:
        txt = fallback
    try:
        hh, mm = map(int, txt.split(":"))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError
        return hh, mm
    except Exception:
        fh, fm = map(int, fallback.split(":"))
        return fh, fm

def send_telegram(text: str, token: str, chat_id: str):
    try:
        import requests
        requests.get(
            f"https://api.telegram.org/bot{token}/sendMessage",
            params={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception:
        pass

# ---------------- Zerodha helpers ----------------

def get_kite_from_env() -> KiteConnect:
    api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise RuntimeError("ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN missing")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    kite.profile()
    print("✅ Zerodha token OK.")
    return kite

INDEX_TOKENS = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
}

LOT_SIZE = {
    "NIFTY": 50,
    "BANKNIFTY": 15,
}

STRIKE_STEP = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
}

def nfo_instruments(kite):
    return kite.instruments("NFO")

def this_week_expiry(dt: datetime) -> date:
    d = dt.date()
    weekday = d.weekday()  # Mon=0 ... Thu=3
    days_ahead = (3 - weekday) % 7
    expiry = d + timedelta(days=days_ahead)
    if weekday == 3 and dt.time() > time(15, 30):
        expiry = expiry + timedelta(days=7)
    return expiry

def find_tokens_for_strike(instr, underlying: str, expiry_d: date, strike: int):
    expiry_d = ensure_date(expiry_d)
    ce = [r for r in instr
          if r.get("name") == underlying
          and ensure_date(r.get("expiry")) == expiry_d
          and int(r.get("strike", 0)) == int(strike)
          and r.get("instrument_type") == "CE"]
    pe = [r for r in instr
          if r.get("name") == underlying
          and ensure_date(r.get("expiry")) == expiry_d
          and int(r.get("strike", 0)) == int(strike)
          and r.get("instrument_type") == "PE"]
    if not ce or not pe:
        raise RuntimeError(f"Could not find CE/PE tokens for {underlying} {expiry_d} @ {strike}")
    return ce[0]["instrument_token"], pe[0]["instrument_token"], ce[0]["tradingsymbol"], pe[0]["tradingsymbol"]

def map_interval(iv: str) -> str:
    return {
        "1m": "minute",
        "3m": "3minute",
        "5m": "5minute",
        "10m": "10minute",
        "15m": "15minute",
    }.get(iv, "5minute")

def fetch_session_ohlc(kite, token: int, for_day: date, interval_alias: str) -> pd.DataFrame:
    start_dt = datetime.combine(for_day, time(9, 15), tzinfo=IST)
    end_dt   = datetime.combine(for_day, time(15, 30), tzinfo=IST)
    candles = kite.historical_data(
        instrument_token=int(token),
        from_date=start_dt,
        to_date=end_dt,
        interval=map_interval(interval_alias),
    )
    df = pd.DataFrame(candles or [])
    if df.empty:
        return df
    df.rename(columns={"date":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def fetch_best_ohlc(kite, token: int, interval_alias: str, prefer_day: date | None) -> tuple[pd.DataFrame, date]:
    """
    1) If prefer_day given, try it; if empty, try previous trading day.
    2) Else try today; if empty, try previous trading day.
    """
    if prefer_day:
        d = prefer_day
        df = fetch_session_ohlc(kite, token, d, interval_alias)
        if not df.empty:
            return df, d
        d2 = prev_trading_day(d)
        df2 = fetch_session_ohlc(kite, token, d2, interval_alias)
        if not df2.empty:
            return df2, d2
        raise RuntimeError(f"No candles for {d} or previous trading day {d2}")

    today = ist_now().date()
    df = fetch_session_ohlc(kite, token, today, interval_alias)
    if not df.empty:
        return df, today
    d2 = prev_trading_day(today)
    df2 = fetch_session_ohlc(kite, token, d2, interval_alias)
    if not df2.empty:
        return df2, d2
    raise RuntimeError("No candles returned for today or previous trading day")

# ---------------- Strategy: ATM Short Straddle ----------------

def make_signal_log(entry_px_ce, entry_px_pe, sl_pct, tgt_pct, lots, lot_size, trail_cfg):
    ce_sl = round(entry_px_ce * (1 + sl_pct/100.0), 2)
    pe_sl = round(entry_px_pe * (1 + sl_pct/100.0), 2)
    pos_value = lots * lot_size
    return {
        "entry": {"CE": entry_px_ce, "PE": entry_px_pe, "qty_per_leg": pos_value},
        "risk":  {"sl_percent_per_leg": sl_pct, "ce_sl": ce_sl, "pe_sl": pe_sl,
                  "combined_target_percent": tgt_pct},
        "trailing": trail_cfg,
    }

# ---------------- Config ----------------

def load_yaml_config(path="config.yaml") -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="./reports")
    args = parser.parse_args()

    cfg = SimpleNamespace(**load_yaml_config())
    iocfg = SimpleNamespace(**cfg.intraday_options)

    kite = get_kite_from_env()

    # Spot & ATM
    index_label = 'NIFTY 50' if iocfg.underlying == 'NIFTY' else 'NIFTY BANK'
    spot_info = kite.ltp([f"NSE:{index_label}"])
    spot_val = list(spot_info.values())[0]["last_price"]
    atm = round_to_strike(spot_val, STRIKE_STEP[iocfg.underlying])
    print(f"ℹ️  {iocfg.underlying} spot {spot_val:.1f} → ATM {atm}")

    # Weekly expiry
    expiry = ensure_date(this_week_expiry(ist_now()))
    print(f"ℹ️  Weekly expiry → {expiry:%Y-%m-%d}")

    # Tokens
    instr = nfo_instruments(kite)
    ce_token, pe_token, ce_sym, pe_sym = find_tokens_for_strike(instr, iocfg.underlying, expiry, atm)
    print(f"✅ Using expiry {expiry:%Y-%m-%d} | CE token: {ce_token} | PE token: {pe_token}")
    print("______________________________________________")

    # Prefer session day from config if provided
    prefer_day = None
    if getattr(iocfg, "date", ""):
        try:
            prefer_day = datetime.strptime(iocfg.date.strip(), "%Y-%m-%d").date()
        except Exception:
            raise RuntimeError(f"Invalid intraday_options.date '{iocfg.date}', use YYYY-MM-DD")

    # OHLC (robust)
    ival = iocfg.interval if hasattr(iocfg, "interval") else "5m"
    ce_df, used_day = fetch_best_ohlc(kite, ce_token, ival, prefer_day)
    pe_df, _       = fetch_best_ohlc(kite, pe_token, ival, prefer_day)
    print(f"📅 Using session: {used_day}")

    # ----- tolerant entry time -----
    raw_entry_time = getattr(iocfg, "entry_time", None) or getattr(iocfg, "start_time", None) or "09:30"
    entry_h, entry_m = parse_hhmm(raw_entry_time, fallback="09:30")

    def first_bar_at(df: pd.DataFrame):
        return df[df["Date"].dt.tz_convert(IST).dt.time >= time(entry_h, entry_m)].head(1)

    ce_bar = first_bar_at(ce_df)
    pe_bar = first_bar_at(pe_df)
    if ce_bar.empty or pe_bar.empty:
        raise RuntimeError("No entry bar found at/after entry_time")

    ce_entry = float(ce_bar["Close"].iloc[0])
    pe_entry = float(pe_bar["Close"].iloc[0])

    lots = int(getattr(iocfg, "lots", 1))
    lot_size = int(getattr(iocfg, "lot_size", LOT_SIZE[iocfg.underlying]))
    sl_pct = float(getattr(iocfg, "leg_sl_percent", 25))
    tgt_pct = float(getattr(iocfg, "combined_target_percent", 0.0))

    trail_cfg = {
        "enabled": bool(getattr(iocfg, "trailing_enabled", False)),
        "type": getattr(iocfg, "trail_type", "atr"),
        "trail_start_atr": float(getattr(iocfg, "trail_start_atr", 1.0)),
        "trail_atr_mult": float(getattr(iocfg, "trail_atr_mult", 1.5)),
        "adx_min": int(getattr(iocfg, "adx_min", 10)),
    }

    log_note = make_signal_log(ce_entry, pe_entry, sl_pct, tgt_pct, lots, lot_size, trail_cfg)

    # Plan print
    print(f"🧾 {ce_sym} | rows: {len(ce_df)} | Close: {ce_entry}")
    print(f"🧾 {pe_sym} | rows: {len(pe_df)} | Close: {pe_entry}")
    print(f"📌 Entry {entry_h:02d}:{entry_m:02d} | Short Straddle {atm} | Lots: {lots} (lot_size {lot_size})")
    print(f"🛡️  SL per leg: {sl_pct}% | 🎯 Combined target: {tgt_pct}% | 🧵 Trailing: {trail_cfg}")

    # Telegram
    t_token = getattr(iocfg, "telegram_bot_token", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    t_chat  = getattr(iocfg, "telegram_chat_id", "") or os.getenv("TELEGRAM_CHAT_ID", "")
    if t_token and t_chat:
        msg = (
            f"🔔 <b>ATM Short Straddle</b>\n"
            f"• <b>{iocfg.underlying}</b> {atm} | {used_day} | Entry {entry_h:02d}:{entry_m:02d} IST\n"
            f"• CE {ce_sym}: {ce_entry}\n"
            f"• PE {pe_sym}: {pe_entry}\n"
            f"• SL/leg: {sl_pct}% | Target: {tgt_pct}%\n"
            f"• Trailing: { 'ON' if trail_cfg['enabled'] else 'OFF' } ({trail_cfg['type']})"
        )
        send_telegram(msg, t_token, t_chat)

    # Report
    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "ts": ist_now().isoformat(),
        "session_date": f"{used_day:%Y-%m-%d}",
        "underlying": iocfg.underlying,
        "atm_strike": atm,
        "expiry": f"{expiry:%Y-%m-%d}",
        "entry_time": f"{entry_h:02d}:{entry_m:02d}",
        "ce": {"symbol": ce_sym, "entry": ce_entry, "token": ce_token},
        "pe": {"symbol": pe_sym, "entry": pe_entry, "token": pe_token},
        "risk": log_note["risk"],
        "trailing": log_note["trailing"],
        "lots": lots,
        "lot_size": lot_size,
    }
    with open(os.path.join(args.out_dir, "latest.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"📦 Wrote {args.out_dir}/latest.json")

if __name__ == "__main__":
    main()
