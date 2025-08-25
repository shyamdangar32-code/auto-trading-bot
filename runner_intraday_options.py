# runner_intraday_options.py
# Intraday ATM Short Straddle (paper) for BANKNIFTY/NIFTY
# - Uses today's intraday window only (fixes "from date after to date")
# - Clean logs (no demo bias)
# - Optional Telegram alerts

import os
import json
import math
import time
import pytz
import yaml
import requests
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd
from kiteconnect import KiteConnect


IST = pytz.timezone("Asia/Kolkata")


# ---------------------------- helpers ----------------------------

def ist_now() -> datetime:
    return datetime.now(IST)


def today_intraday_window() -> Tuple[datetime, datetime]:
    """Return today's intraday window 09:15 → min(now, 15:30) in IST."""
    now = ist_now()
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    day_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    end = min(now, day_end)
    # in case this runs before 09:15, clamp end to start to avoid invalid range
    if end < start:
        end = start + timedelta(minutes=1)
    return start, end


def round_to_step(value: float, step: int) -> int:
    return int(round(value / step) * step)


def next_weekly_expiry(from_dt: Optional[datetime] = None) -> datetime:
    """Return upcoming Thursday (weekly expiry) in IST (date only)."""
    d = from_dt.astimezone(IST) if from_dt else ist_now()
    # if it's Thursday after 3:30pm, move to next week
    weekday = d.weekday()  # Mon=0 ... Sun=6
    days_ahead = (3 - weekday) % 7  # Thursday = 3
    expiry = (d + timedelta(days=days_ahead)).date()
    if weekday == 3 and d.time() > datetime(1,1,1,15,30).time():
        expiry = (d + timedelta(days=7)).date()
    return datetime.combine(expiry, datetime.min.time()).replace(tzinfo=IST)


def telegram_notify(token: str, chat_id: str, text: str) -> None:
    """Fire-and-forget Telegram message; silence any network error."""
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_kite() -> KiteConnect:
    api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    access = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise RuntimeError("ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN missing in env.")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    # sanity
    prof = kite.profile()
    print(f"✅ Zerodha token OK. User: {prof.get('user_id')} | {prof.get('user_name')}")
    return kite


def get_spot(kite: KiteConnect, underlying: str) -> float:
    """Get spot for BANKNIFTY / NIFTY index via LTP."""
    symbol_map = {
        "BANKNIFTY": "NSE:NIFTY BANK",
        "NIFTY": "NSE:NIFTY 50",
    }
    key = symbol_map.get(underlying.upper())
    if not key:
        raise ValueError("underlying must be NIFTY or BANKNIFTY")
    ltp = kite.ltp([key])
    return float(ltp[key]["last_price"])


def pick_atm_tokens(
    kite: KiteConnect,
    underlying: str,
    expiry_dt: datetime,
    strike: int
) -> Tuple[int, int, str, str]:
    """Return (CE_token, PE_token, CE_symbol, PE_symbol) for given underlying/expiry/strike."""
    # Pull once; filter in memory
    nfo = pd.DataFrame(kite.instruments("NFO"))
    nfo = nfo[(nfo["name"] == underlying.upper()) & (nfo["segment"] == "NFO-OPT")]
    # expiry is timezone-naive date in exchange listing; compare on date
    exp_date = expiry_dt.date()
    nfo = nfo[nfo["expiry"].dt.date == exp_date]
    nfo = nfo[nfo["strike"] == float(strike)]

    ce = nfo[nfo["instrument_type"] == "CE"].head(1)
    pe = nfo[nfo["instrument_type"] == "PE"].head(1)
    if ce.empty or pe.empty:
        raise RuntimeError(f"Could not find CE/PE tokens for {underlying} {exp_date} @ {strike}")

    ce_token = int(ce.iloc[0]["instrument_token"])
    pe_token = int(pe.iloc[0]["instrument_token"])
    return ce_token, pe_token, ce.iloc[0]["tradingsymbol"], pe.iloc[0]["tradingsymbol"]


def get_hist(kite: KiteConnect, token: int, from_dt: datetime, to_dt: datetime, interval: str) -> pd.DataFrame:
    """Fetch intraday OHLCV for today window, robust to Zerodha rules."""
    # Kite expects timezone-naive datetimes; pass IST-naive
    f = from_dt.astimezone(IST).replace(tzinfo=None)
    t = to_dt.astimezone(IST).replace(tzinfo=None)
    if f >= t:
        # guardrail to avoid "from > to"
        t = f + timedelta(minutes=1)
    data = kite.historical_data(token, f, t, interval)
    df = pd.DataFrame(data)
    if df.empty:
        raise RuntimeError("No candles returned")
    df.rename(columns={"date": "Date", "open": "Open", "high": "High",
                      "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------------------------- strategy (paper) ----------------------------

@dataclass
class Leg:
    symbol: str
    token: int
    side: str  # SELL
    entry_price: float
    sl_price: float
    trail_active: bool = False


def simulate_short_straddle(
    ce_df: pd.DataFrame,
    pe_df: pd.DataFrame,
    entry_time_str: str,
    leg_sl_percent: float,
    combined_target_percent: Optional[float],
    trailing: bool,
) -> Dict:
    """
    Paper-sim only:
      - Entry: first candle with Date.time() >= entry_time
      - SL: % of entry price (per leg)
      - Optional combined MTM target
      - Trailing: not implemented deeply; placeholder switch kept for logs
    """
    entry_time = datetime.strptime(entry_time_str, "%H:%M").time()

    ce_entry_row = ce_df[ce_df["Date"].dt.time >= entry_time].head(1)
    pe_entry_row = pe_df[pe_df["Date"].dt.time >= entry_time].head(1)
    if ce_entry_row.empty or pe_entry_row.empty:
        return {"note": "No entry candle ≥ entry_time", "trades": []}

    ce_entry = float(ce_entry_row.iloc[0]["Close"])
    pe_entry = float(pe_entry_row.iloc[0]["Close"])

    ce_sl = ce_entry * (1 + leg_sl_percent / 100.0)
    pe_sl = pe_entry * (1 + leg_sl_percent / 100.0)

    # For a simple paper preview we won’t walk bar-by-bar exits;
    # we’ll just produce entry + SL + optional combined target levels.
    result = {
        "entry_time": str(ce_entry_row.iloc[0]["Date"]),
        "legs": [
            {"symbol": "CE", "side": "SELL", "entry": ce_entry, "sl": ce_sl},
            {"symbol": "PE", "side": "SELL", "entry": pe_entry, "sl": pe_sl},
        ],
        "combined_target": None,
        "trailing_enabled": bool(trailing),
    }
    if combined_target_percent is not None:
        total_entry = ce_entry + pe_entry
        target_value = total_entry * (1 - combined_target_percent / 100.0)
        result["combined_target"] = target_value
    return result


# ---------------------------- main ----------------------------

def main():
    cfg = load_config()

    intr = cfg.get("intraday_options", {}) or {}
    underlying = intr.get("underlying", "BANKNIFTY").upper()
    interval = intr.get("interval", "5minute")
    entry_time = intr.get("entry_time", "09:30")
    leg_sl_percent = float(intr.get("leg_sl_percent", 30.0))
    combined_target_percent = intr.get("combined_target_percent", None)
    trailing = bool(intr.get("trail", False)) or bool(intr.get("trailing_enabled", False))
    lots = int(intr.get("lots", 1))
    lot_size = int(intr.get("lot_size", 15 if underlying == "BANKNIFTY" else 50))
    out_dir = cfg.get("out_dir", "reports")

    tg_token = (intr.get("telegram_bot_token") or "").strip()
    tg_chat = (intr.get("telegram_chat_id") or "").strip()

    os.makedirs(out_dir, exist_ok=True)

    kite = get_kite()

    # 1) Spot & ATM strike
    spot = get_spot(kite, underlying)
    step = 100 if underlying == "BANKNIFTY" else 50
    atm = round_to_step(spot, step)
    print(f"ℹ️  {underlying} spot {spot:.1f} → ATM {atm}")

    # 2) Weekly expiry (date only) – used for picking option tokens, NOT for history range
    expiry_dt = next_weekly_expiry()
    print(f"ℹ️  Weekly expiry → {expiry_dt.date()}")

    ce_token, pe_token, ce_sym, pe_sym = pick_atm_tokens(kite, underlying, expiry_dt, atm)
    print(f"✅ Using expiry {expiry_dt.date()} | CE token: {ce_token} | PE token: {pe_token}")
    print("—" * 60)

    # 3) Today’s intraday window (fix for 'from date after to date')
    day_start, day_end = today_intraday_window()

    # 4) Pull OHLC
    ce_df = get_hist(kite, ce_token, day_start, day_end, interval)
    pe_df = get_hist(kite, pe_token, day_start, day_end, interval)

    # 5) Simulate paper short straddle (clean preview)
    res = simulate_short_straddle(
        ce_df, pe_df,
        entry_time_str=entry_time,
        leg_sl_percent=leg_sl_percent,
        combined_target_percent=combined_target_percent,
        trailing=trailing
    )

    if not res.get("legs"):
        print("ℹ️  No entry candle >= entry_time; nothing to do.")
    else:
        print(f"🕒 Entry @ {res['entry_time']}")
        for leg in res["legs"]:
            print(f"🔻 SELL {leg['symbol']}  entry={leg['entry']:.2f}  SL={leg['sl']:.2f}")
        if res.get("combined_target") is not None:
            print(f"🎯 Combined target (premium) = {res['combined_target']:.2f}")
        if res.get("trailing_enabled"):
            print("📍 Trailing: ENABLED (ATR/percent logic can be extended here)")
        else:
            print("📍 Trailing: disabled")

        # Telegram alert (optional)
        msg_lines = [
            f"<b>{underlying} ATM Short Straddle (paper)</b>",
            f"Expiry: {expiry_dt.date()}   Lots: {lots} (lot_size {lot_size})",
            f"Entry time: {entry_time}",
            f"CE entry: {res['legs'][0]['entry']:.2f}  SL: {res['legs'][0]['sl']:.2f}",
            f"PE entry: {res['legs'][1]['entry']:.2f}  SL: {res['legs'][1]['sl']:.2f}",
        ]
        if res.get("combined_target") is not None:
            msg_lines.append(f"Target (combined premium): {res['combined_target']:.2f}")
        if res.get("trailing_enabled"):
            msg_lines.append("Trailing: ENABLED")
        telegram_notify(tg_token, tg_chat, "\n".join(msg_lines))

    # 6) Write artifact json for inspection
    artifact = {
        "underlying": underlying,
        "spot": spot,
        "atm": atm,
        "expiry": str(expiry_dt.date()),
        "entry_time": res.get("entry_time"),
        "legs": res.get("legs", []),
        "combined_target": res.get("combined_target"),
        "trailing_enabled": res.get("trailing_enabled"),
    }
    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"📦 Wrote {out_dir}/latest.json")


if __name__ == "__main__":
    main()
