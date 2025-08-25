# runner_intraday_options.py
# Intraday options (paper) — ATM short straddle with SL / target / ATR trailing
# - Zerodha-only (expects ZERODHA_API_KEY + ZERODHA_ACCESS_TOKEN in env)
# - Config is read from ./config.yaml
# - Writes ./reports/latest.json and logs human-friendly lines
# - Sends Telegram alerts if configured

import os
import json
import math
from datetime import datetime, date, time, timedelta, timezone

import pytz
import pandas as pd
import yaml
from kiteconnect import KiteConnect

# ----------------------- small helpers -----------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ist_now():
    return datetime.now(pytz.timezone("Asia/Kolkata"))

def send_telegram(bot_token: str, chat_id: str, text: str):
    """Fire-and-forget Telegram message (silent if not configured)."""
    if not bot_token or not chat_id:
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        requests.post(url, json=payload, timeout=10)
    except Exception as _:
        # keep CI green even if telegram fails
        pass

def get_kite_from_env() -> KiteConnect:
    api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise RuntimeError("Missing ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN in env.")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    # quick sanity
    kite.profile()
    print("✅ Zerodha token OK.")
    return kite

def next_weekly_expiry(d: date) -> date:
    # India weekly index options expire on Thursday
    # If today is Thursday and before close, use today; else next Thursday
    weekday = d.weekday()  # Mon=0 ... Sun=6, Thu=3
    days_ahead = (3 - weekday) % 7
    if days_ahead == 0:
        return d
    return d + timedelta(days=days_ahead)

def floor_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # df has columns: Open, High, Low, Close
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

# ----------------------- instrument discovery -----------------------

INDEX_TOKENS = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
}

STRIKE_STEP = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
}

def get_spot_ltp(kite: KiteConnect, underlying: str) -> float:
    token = INDEX_TOKENS[underlying]
    # ltp can accept numeric instrument tokens
    data = kite.ltp([token])
    # key is the same token converted to string
    key = str(token)
    return float(data[key]["last_price"])

def find_weekly_tokens(kite: KiteConnect, underlying: str, expiry_dt: date, strike: int):
    """Return (ce_token, pe_token) for the weekly options."""
    nfo = kite.instruments("NFO")
    ce_token = pe_token = None
    for ins in nfo:
        if ins.get("segment") != "NFO-OPT":
            continue
        if ins.get("name") != underlying:
            continue
        # match weekly expiry (no explicit monthly check; good enough for paper)
        if pd.to_datetime(ins.get("expiry")).date() != expiry_dt:
            continue
        if int(ins.get("strike")) != int(strike):
            continue
        if ins.get("instrument_type") == "CE":
            ce_token = ins.get("instrument_token")
        elif ins.get("instrument_type") == "PE":
            pe_token = ins.get("instrument_token")
    if not ce_token or not pe_token:
        raise RuntimeError(f"Could not find CE/PE tokens for {underlying} {expiry_dt} @ {strike}")
    return int(ce_token), int(pe_token)

def get_hist(kite: KiteConnect, token: int, from_dt: datetime, to_dt: datetime, interval: str):
    candles = kite.historical_data(
        instrument_token=token,
        from_date=from_dt,
        to_date=to_dt,
        interval=interval,
    )
    if not candles:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    df = pd.DataFrame(candles)
    df.rename(columns={"date":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date","Open","High","Low","Close","Volume"]]

# ----------------------- main engine -----------------------

def main():
    cfg = load_config("config.yaml")

    tz = pytz.timezone(cfg.get("tz", "Asia/Kolkata"))
    io_cfg = cfg.get("intraday_options", {})
    underlying = io_cfg.get("underlying", "BANKNIFTY").upper()
    lots       = int(io_cfg.get("lots", 1))
    lot_size   = int(io_cfg.get("lot_size", 15))
    ival       = io_cfg.get("interval", "5minute")
    entry_rule = io_cfg.get("entry_rule", "atm_short_straddle")

    entry_time_str = io_cfg.get("entry_time", "09:30")
    squareoff_str  = io_cfg.get("squareoff", "15:10")
    date_override  = (io_cfg.get("date") or "").strip()

    leg_sl_pct     = float(io_cfg.get("leg_sl_percent", 30.0))
    comb_tgt_pct   = float(io_cfg.get("combined_target_percent", 50.0))
    trail_on       = bool(io_cfg.get("trail", False) or io_cfg.get("trailing_enabled", False))
    trail_start_atr= float(io_cfg.get("trail_start_atr", 1.0))
    trail_atr_mult = float(io_cfg.get("trail_atr_mult", 1.5))
    adx_min        = float(io_cfg.get("adx_min", 10))

    # Telegram
    tg_token = io_cfg.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TG_BOT_TOKEN") or ""
    tg_chat  = io_cfg.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("TG_CHAT_ID") or ""

    # Trading day
    if date_override:
        trade_day = datetime.strptime(date_override, "%Y-%m-%d").date()
    else:
        trade_day = ist_now().date()

    # Entry / squareoff datetimes (IST)
    entry_dt    = tz.localize(datetime.combine(trade_day, datetime.strptime(entry_time_str, "%H:%M").time()))
    squareoff_dt= tz.localize(datetime.combine(trade_day, datetime.strptime(squareoff_str, "%H:%M").time()))
    now_ist     = ist_now()

    # Kite
    kite = get_kite_from_env()

    # 1) Discover ATM
    spot = get_spot_ltp(kite, underlying)
    step = STRIKE_STEP[underlying]
    atm_strike = floor_to_step(spot, step)
    print(f"ℹ️  {underlying} spot {spot:.1f} → ATM {atm_strike}")

    # 2) Expiry
    expiry = next_weekly_expiry(trade_day)
    print(f"ℹ️  Weekly expiry → {expiry}")

    # 3) CE/PE instrument tokens
    ce_token, pe_token = find_weekly_tokens(kite, underlying, expiry, atm_strike)
    print(f"✅ Using expiry {expiry} | CE token: {ce_token} | PE token: {pe_token}")
    print("—" * 56)

    # 4) History window (for the day)
    day_start = tz.localize(datetime.combine(trade_day, time(9, 15)))
    to_dt     = min(now_ist, squareoff_dt)

    ce_df = get_hist(kite, ce_token, day_start, to_dt, ival)
    pe_df = get_hist(kite, pe_token, day_start, to_dt, ival)

    # If there is no candle yet (very early), exit gracefully
    if ce_df.empty or pe_df.empty:
        print("ℹ️  No candles yet for options; exiting.")
        return

    # 5) Find entry bar close at/after entry_time
    ce_entry_row = ce_df[ce_df["Date"] >= entry_dt].head(1)
    pe_entry_row = pe_df[pe_df["Date"] >= entry_dt].head(1)
    if ce_entry_row.empty or pe_entry_row.empty:
        print("ℹ️  Entry time not reached yet; exiting.")
        return

    ce_entry_px = float(ce_entry_row["Close"].iloc[0])
    pe_entry_px = float(pe_entry_row["Close"].iloc[0])
    entry_bar_time = ce_entry_row["Date"].iloc[0].to_pydatetime().astimezone(tz)

    # 6) Risk math
    ce_sl = round(ce_entry_px * (1 + leg_sl_pct/100.0), 2)   # short → SL above entry
    pe_sl = round(pe_entry_px * (1 + leg_sl_pct/100.0), 2)
    comb_entry = ce_entry_px + pe_entry_px
    comb_target_val = round(comb_entry * (1 - comb_tgt_pct/100.0), 2)

    # Trailing (ATR on each leg)
    ce_df["ATR"] = atr(ce_df)
    pe_df["ATR"] = atr(pe_df)
    ce_last_atr = float(ce_df["ATR"].iloc[-1])
    pe_last_atr = float(pe_df["ATR"].iloc[-1])

    trailing_info = None
    if trail_on:
        ce_trail_dist = round(trail_atr_mult * ce_last_atr, 2)
        pe_trail_dist = round(trail_atr_mult * pe_last_atr, 2)
        trailing_info = {
            "start_trigger_atr": trail_start_atr,
            "ce_atr": ce_last_atr,
            "pe_atr": pe_last_atr,
            "ce_trail_dist": ce_trail_dist,
            "pe_trail_dist": pe_trail_dist,
        }

    # 7) Present state @ latest bar
    ce_last = float(ce_df["Close"].iloc[-1])
    pe_last = float(pe_df["Close"].iloc[-1])
    comb_last = ce_last + pe_last

    # 8) Logs
    print(f"🕘 Entry bar (>= {entry_time_str}) @ {entry_bar_time.strftime('%H:%M')}")
    print(f"🎯 Strategy: {entry_rule} | Lots={lots} | Lot size={lot_size}")
    print(f"🟠 SELL CE @ {ce_entry_px:.2f} | SL: {ce_sl:.2f}")
    print(f"🟠 SELL PE @ {pe_entry_px:.2f} | SL: {pe_sl:.2f}")
    print(f"📉 Combined entry premium: {comb_entry:.2f}")
    print(f"✅ Combined target ({comb_tgt_pct:.0f}%): {comb_target_val:.2f}")
    if trailing_info:
        print(f"🔁 Trailing ON (ATR): start>{trail_start_atr}*ATR | CE trail≈{trailing_info['ce_trail_dist']:.2f} | PE trail≈{trailing_info['pe_trail_dist']:.2f}")
    else:
        print("🔁 Trailing: OFF")

    print("—" * 56)
    print(f"📊 Latest bar: CE={ce_last:.2f} | PE={pe_last:.2f} | Combo={comb_last:.2f}")
    print(f"⏹️  Auto squareoff: {squareoff_str} IST")

    # 9) Telegram push
    msg = (
        f"📈 {underlying} {expiry} ATM Short Straddle\n"
        f"Entry@{entry_bar_time.strftime('%H:%M')}  CE {atm_strike} @ {ce_entry_px:.2f}  | SL {ce_sl:.2f}\n"
        f"                          PE {atm_strike} @ {pe_entry_px:.2f}  | SL {pe_sl:.2f}\n"
        f"Target (combo −{comb_tgt_pct:.0f}%): {comb_target_val:.2f}  | Latest combo: {comb_last:.2f}\n"
        f"Trailing: {'ON' if trailing_info else 'OFF'}  | Squareoff: {squareoff_str}"
    )
    send_telegram(tg_token, tg_chat, msg)

    # 10) Persist a JSON report
    os.makedirs("reports", exist_ok=True)
    out = {
        "generated_at": ist_now().isoformat(),
        "underlying": underlying,
        "expiry": str(expiry),
        "atm_strike": atm_strike,
        "entry_time": entry_bar_time.strftime("%H:%M"),
        "legs": {
            "CE": {"token": ce_token, "entry": ce_entry_px, "sl": ce_sl, "last": ce_last},
            "PE": {"token": pe_token, "entry": pe_entry_px, "sl": pe_sl, "last": pe_last},
        },
        "combined": {
            "entry": comb_entry,
            "last": comb_last,
            "target_percent": comb_tgt_pct,
            "target_value": comb_target_val,
        },
        "trailing": trailing_info or {"enabled": False},
        "squareoff": squareoff_str,
        "interval": ival,
        "lots": lots,
        "lot_size": lot_size,
    }
    with open("reports/latest.json", "w") as f:
        json.dump(out, f, indent=2)
    print("🗂️  Wrote reports/latest.json")

if __name__ == "__main__":
    main()
