#!/usr/bin/env python3
# tools/weekly_opt.py
#
# Weekly optimizer / backtest runner using Zerodha OHLC (no Yahoo).
# - Reads inputs from env vars (see names below).
# - Fetches OHLC for NIFTY/BANKNIFTY from Kite.
# - Runs bot.backtest.run_backtest(prices, cfg)
# - Writes reports to ./reports (metrics.json, trades.csv, equity.csv)
# - Optionally posts a Telegram summary.

import os
import json
from datetime import datetime, timedelta, timezone, time

import pandas as pd
from kiteconnect import KiteConnect

# ---- Repo modules
from bot.backtest import run_backtest, save_reports

# ---------------- Time/market helpers ----------------
IST = timezone(timedelta(hours=5, minutes=30))

INDEX_TOKENS = {
    "NIFTY": 256265,      # NSE Index
    "BANKNIFTY": 260105,  # NSE Bank Index
}

def ist_dt(y, m, d, hh=0, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=IST)

def parse_date_yyyy_mm_dd(txt: str) -> datetime:
    dt = datetime.strptime(txt, "%Y-%m-%d").date()
    # fetch the full day window in IST
    return ist_dt(dt.year, dt.month, dt.day)

# ---------------- Kite helpers ----------------
def get_kite_from_env() -> KiteConnect:
    api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise RuntimeError("ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN missing")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    # sanity (throws if invalid)
    kite.profile()
    return kite

def fetch_zerodha_ohlc(kite: KiteConnect, token: int,
                       start: datetime, end: datetime,
                       interval: str = "day") -> pd.DataFrame:
    candles = kite.historical_data(
        instrument_token=int(token),
        from_date=start,
        to_date=end,
        interval=interval,
    )
    df = pd.DataFrame(candles or [])
    if df.empty:
        raise RuntimeError(f"No OHLC from Zerodha for token={token} in range {start} → {end}")

    df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low":  "Low",
        "close":"Close",
        "volume":"Volume",
    }, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# ---------------- Telegram ----------------
def send_telegram(text: str, token: str, chat_id: str):
    if not token or not chat_id:
        return
    try:
        import requests
        requests.get(
            f"https://api.telegram.org/bot{token}/sendMessage",
            params={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception:
        # best-effort only
        pass

# ---------------- Main ----------------
def getenv_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")

def main():
    # ---- Inputs from workflow (ENV). Provide sensible defaults.
    INDEX       = (os.getenv("INDEX") or "BANKNIFTY").strip().upper()   # NIFTY or BANKNIFTY
    START_DATE  = (os.getenv("START_DATE") or "").strip()               # YYYY-MM-DD
    END_DATE    = (os.getenv("END_DATE") or "").strip()                 # YYYY-MM-DD
    INTERVAL    = (os.getenv("INTERVAL") or "day").strip().lower()      # 'day', 'minute', '5minute', etc.

    CAPITAL_RS  = float(os.getenv("CAPITAL", "100000"))
    ORDER_QTY   = int(os.getenv("ORDER_QTY", "1"))
    STOP_PCT    = float(os.getenv("STOP_PCT", "25"))
    TARGET_PCT  = float(os.getenv("TARGET_PCT", "0"))
    RE_MAX      = int(os.getenv("REENTRY_MAX", "0"))
    COOLDOWN    = int(os.getenv("COOLDOWN", "0"))

    TRAIL_ON    = getenv_bool("TRAILING_ENABLED", False)
    TRAIL_TYPE  = (os.getenv("TRAIL_TYPE") or "atr").strip()
    ATR_MULT    = float(os.getenv("ATR_MULT", "1.5"))
    ADX_MIN     = int(os.getenv("ADX_MIN", "10"))

    OUT_DIR     = (os.getenv("OUT_DIR") or "./reports").strip()

    T_TOKEN     = os.getenv("TELEGRAM_BOT_TOKEN", "")
    T_CHAT      = os.getenv("TELEGRAM_CHAT_ID", "")
    T_SUMMARY   = getenv_bool("TELEGRAM_SUMMARY", True)

    if INDEX not in INDEX_TOKENS:
        raise RuntimeError(f"INDEX must be NIFTY or BANKNIFTY; got {INDEX}")

    if not START_DATE or not END_DATE:
        raise RuntimeError("Provide START_DATE and END_DATE as YYYY-MM-DD")

    start = parse_date_yyyy_mm_dd(START_DATE)
    end   = parse_date_yyyy_mm_dd(END_DATE) + timedelta(hours=23, minutes=59, seconds=59)

    # ---- Kite and data
    kite = get_kite_from_env()
    print(f"✅ Zerodha auth OK | Fetching {INDEX} {INTERVAL} OHLC {START_DATE} → {END_DATE}")

    token = INDEX_TOKENS[INDEX]
    prices = fetch_zerodha_ohlc(kite, token, start, end, interval=INTERVAL)

    # ---- Backtest config expected by bot.strategy / bot.backtest
    cfg = {
        "order_qty": ORDER_QTY,
        "capital_rs": CAPITAL_RS,
        "reentry_max": RE_MAX,
        "reentry_cooldown": COOLDOWN,

        # risk & exits
        "stoploss_pct": STOP_PCT,              # per-leg SL %
        "target_pct": TARGET_PCT,              # combined target %

        # trailing
        "trailing_enabled": TRAIL_ON,
        "trail_type": TRAIL_TYPE,              # 'atr'
        "trail_start_atr": 1.0,                # can be tuned if you expose input
        "trail_atr_mult": ATR_MULT,
        "adx_min": ADX_MIN,
    }

    # ---- Run backtest
    summary, trades_df, equity_ser = run_backtest(prices, cfg)

    # ---- Write reports
    os.makedirs(OUT_DIR, exist_ok=True)
    save_reports(OUT_DIR, summary, trades_df, equity_ser)

    print("______________________________________________")
    print(f"📊 Summary | {INDEX} {START_DATE} → {END_DATE}")
    print(json.dumps(summary, indent=2))
    print(f"🗂  Reports written to: {OUT_DIR}/(metrics.json, trades.csv, equity.csv)")

    # ---- Telegram summary (optional)
    if T_SUMMARY and T_TOKEN and T_CHAT:
        msg = (
            f"📈 <b>Weekly Backtest</b>\n"
            f"• <b>{INDEX}</b> {START_DATE} → {END_DATE}\n"
            f"• Qty: {ORDER_QTY} | Capital: ₹{int(CAPITAL_RS):,}\n"
            f"• SL: {STOP_PCT}% | Target: {TARGET_PCT}% | Trail: {'ON' if TRAIL_ON else 'OFF'}\n"
            f"• Trades: {summary.get('n_trades', 0)} | Win%: {summary.get('win_rate', 0)}%\n"
            f"• ROI: {summary.get('roi_pct', 0)}% | Max DD: {summary.get('max_dd_pct', 0)}%\n"
            f"• Time DD (bars): {summary.get('time_dd_bars', 0)} | R:R: {summary.get('rr', 0)}"
        )
        send_telegram(msg, T_TOKEN, T_CHAT)

if __name__ == "__main__":
    main()
