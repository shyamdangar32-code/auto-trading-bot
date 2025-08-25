# runner_intraday_options.py
# Intraday ATM short-straddle (paper) for BANKNIFTY/NIFTY with real logs + Telegram

import os
import sys
import json
import math
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np

from kiteconnect import KiteConnect
from bot.config import load_config   # <- your existing helper


# ------------------- Utils -------------------

IST = "Asia/Kolkata"

def tz_now_ist():
    # naive -> local IST time using pandas (no extra deps)
    return pd.Timestamp.now(tz=IST).to_pydatetime()

def send_telegram(msg: str):
    """Fire-and-forget Telegram message if envs are present."""
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat  = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat:
        return
    import urllib.parse, urllib.request
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat, "text": msg, "parse_mode": "HTML"}).encode()
    try:
        urllib.request.urlopen(url, data=payload, timeout=10)
    except Exception:
        pass


def round_to_strike(x: float, step: int) -> int:
    return int(round(x / step) * step)


def weekly_thursday(from_dt: datetime) -> date:
    """Next/this Thursday for weekly options (including today if Thu)."""
    d = from_dt.date()
    offset = (3 - d.weekday()) % 7  # Thu = 3
    return d + timedelta(days=offset)


def get_kite() -> KiteConnect:
    ak = (os.getenv("ZERODHA_API_KEY") or "").strip()
    at = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not ak or not at:
        raise RuntimeError("ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN missing")
    kite = KiteConnect(api_key=ak)
    kite.set_access_token(at)
    # quick check
    _ = kite.profile()
    return kite


def get_hist(kite: KiteConnect, token: int, frm: datetime, to: datetime, interval: str) -> pd.DataFrame:
    """
    SAFE historical fetch that guarantees a datetime column.
    Fixes: ensure `date` is converted to datetime so .dt works.
    """
    candles = kite.historical_data(token, frm, to, interval)
    df = pd.DataFrame(candles)
    if df.empty:
        return df
    # Standardize and ensure datetime dtype
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    # Rename for consistency
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    return df


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


# ------------------- Strategy bits -------------------

@dataclass
class Leg:
    side: str      # "CE" or "PE"
    token: int
    entry: float = np.nan
    sl_price: float = np.nan
    trail_active: bool = False


def select_tokens(kite: KiteConnect, underlying: str, expiry_d: date, atm_strike: int):
    """
    Find CE/PE instrument tokens for the given underlying/expiry/strike.
    """
    seg = "NFO"
    ins = kite.instruments(seg)
    ul_prefix = "BANKNIFTY" if underlying.upper().startswith("BANK") else "NIFTY"
    rows = [r for r in ins if r["segment"] == "NFO-OPT"
            and r["tradingsymbol"].startswith(ul_prefix)
            and r["expiry"].date() == expiry_d
            and int(r["strike"]) == atm_strike]
    ce = next((r for r in rows if r["instrument_type"] == "CE"), None)
    pe = next((r for r in rows if r["instrument_type"] == "PE"), None)
    if not ce or not pe:
        raise RuntimeError(f"Could not find CE/PE tokens for {ul_prefix} {expiry_d} @ {atm_strike}")
    return ce["instrument_token"], pe["instrument_token"], ce["tradingsymbol"], pe["tradingsymbol"]


def ltp(kite: KiteConnect, instrument_token: int) -> float:
    q = kite.ltp([instrument_token])
    key = str(instrument_token)
    if instrument_token in q:
        return q[instrument_token]["last_price"]
    if key in q:
        return q[key]["last_price"]
    # Fallback via quote
    x = kite.quote([instrument_token])
    if instrument_token in x:
        return x[instrument_token]["last_price"]
    if key in x:
        return x[key]["last_price"]
    raise RuntimeError("LTP not available")


# ------------------- Runner -------------------

def run_intraday(config_path: str, out_dir: str):
    cfg = load_config(config_path)

    intr = cfg.get("intraday_options", {})
    underlying = intr.get("underlying", "BANKNIFTY").upper()
    lots        = int(intr.get("lots", 1))
    lot_size    = int(intr.get("lot_size", 15 if underlying == "BANKNIFTY" else 50))
    entry_time  = intr.get("entry_time", "09:30")
    squareoff   = intr.get("squareoff", "15:10")
    sim_date_s  = (intr.get("date") or "").strip()
    interval    = intr.get("interval", "5minute")
    leg_sl_pct  = float(intr.get("leg_sl_percent", 30.0))
    combo_tgt   = float(intr.get("combined_target_percent", 50.0))
    trailing_on = bool(intr.get("trailing_enabled", True))
    trail_type  = intr.get("trail_type", "atr")
    trail_start = float(intr.get("trail_start_atr", 1.0))
    trail_mult  = float(intr.get("trail_atr_mult", 1.5))
    adx_min     = float(intr.get("adx_min", 10))

    os.makedirs(out_dir, exist_ok=True)

    kite = get_kite()
    print("✅ Zerodha token OK.")

    # ---- Spot & ATM strike
    # BankNifty index token (official) is 260105; NIFTY is 256265
    idx_token = 260105 if underlying == "BANKNIFTY" else 256265
    spot = ltp(kite, idx_token)
    strike_step = 100 if underlying == "BANKNIFTY" else 50
    atm = round_to_strike(spot, strike_step)
    print(f"ℹ️  {underlying} spot {spot:.1f} → ATM {atm}")

    # ---- Expiry
    now = tz_now_ist()
    if sim_date_s:
        # simulate as if "today" = sim_date
        try:
            sim_dt = datetime.strptime(sim_date_s, "%Y-%m-%d")
        except ValueError:
            raise RuntimeError("intraday_options.date must be YYYY-MM-DD")
        expiry_d = weekly_thursday(sim_dt)
    else:
        expiry_d = weekly_thursday(now)
    print(f"ℹ️  Weekly expiry → {expiry_d}")

    # ---- Pick CE/PE tokens
    ce_token, pe_token, ce_sym, pe_sym = select_tokens(kite, underlying, expiry_d, atm)
    print(f"✅ Using expiry {expiry_d} | CE token: {ce_token} | PE token: {pe_token}")

    # ---- Historical candles (today)
    # from market open to now for context; Zerodha requires from <= to
    day_start = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=9, minutes=15)
    to_dt = now
    ce_df = get_hist(kite, ce_token, day_start, to_dt, interval)
    pe_df = get_hist(kite, pe_token, day_start, to_dt, interval)

    # Ensure we have candles
    if ce_df.empty or pe_df.empty:
        raise RuntimeError("No candles for CE/PE — market holiday or bad token?")

    # Compute ATR on underlying legs individually for trailing, if requested
    ce_df["ATR"] = atr(ce_df, 14)
    pe_df["ATR"] = atr(pe_df, 14)

    print("—" * 60)
    print(f"📈 {ce_sym} | rows: {len(ce_df)} | Close: {float(ce_df['Close'].iloc[-1]):.2f}")
    print(f"📈 {pe_sym} | rows: {len(pe_df)} | Close: {float(pe_df['Close'].iloc[-1]):.2f}")

    # ---- Simulated entry at latest completed bar at/after entry_time
    # Safe .dt usage (we fixed dtype in get_hist)
    et_hours, et_minutes = map(int, entry_time.split(":"))
    entry_mask = (ce_df["date"].dt.hour > et_hours) | ((ce_df["date"].dt.hour == et_hours) & (ce_df["date"].dt.minute >= et_minutes))
    if not entry_mask.any():
        raise RuntimeError("No bar at/after entry_time yet.")
    entry_idx = np.where(entry_mask)[0][0]

    ce_entry = float(ce_df["Close"].iloc[entry_idx])
    pe_entry = float(pe_df["Close"].iloc[entry_idx])

    ce_leg = Leg(side="CE", token=ce_token, entry=ce_entry)
    pe_leg = Leg(side="PE", token=pe_token, entry=pe_entry)

    ce_leg.sl_price = round(ce_leg.entry * (1 + leg_sl_pct / 100.0), 2)  # SL for short position (price rises)
    pe_leg.sl_price = round(pe_leg.entry * (1 + leg_sl_pct / 100.0), 2)

    combo_entry = ce_leg.entry + pe_leg.entry
    combo_tgt_val = round(combo_entry * (1 - combo_tgt / 100.0), 2)

    header = (
        f"🚀 <b>Paper SELL ATM Straddle</b> ({underlying})\n"
        f"• ATM {atm} | Exp {expiry_d}\n"
        f"• Lots {lots} × lot_size {lot_size}\n"
        f"• Entry {entry_time} bar: CE={ce_leg.entry:.2f}, PE={pe_leg.entry:.2f} (Combo={combo_entry:.2f})\n"
        f"• SL per leg {leg_sl_pct:.0f}% → CE SL {ce_leg.sl_price:.2f}, PE SL {pe_leg.sl_price:.2f}\n"
        f"• Combo target {combo_tgt:.0f}% → {combo_tgt_val:.2f}\n"
        f"• Trailing: {'ON' if trailing_on else 'OFF'} ({trail_type}, start>={trail_start}×ATR, dist={trail_mult}×ATR)"
    )
    print(header)
    send_telegram(header)

    # ---- Paper loop over remaining bars until squareoff
    so_hours, so_minutes = map(int, squareoff.split(":"))
    exit_time = datetime.combine(now.date(), datetime.min.time()) + timedelta(hours=so_hours, minutes=so_minutes)

    pnl_rows = []
    ce_sl_hit = pe_sl_hit = False
    combo_hit = False

    for i in range(entry_idx + 1, len(ce_df)):
        bar_dt = ce_df["date"].iloc[i].to_pydatetime()
        if bar_dt >= exit_time:
            print(f"🕒 Squareoff reached @ {bar_dt.time()}")
            break

        ce_close = float(ce_df["Close"].iloc[i])
        pe_close = float(pe_df["Close"].iloc[i])

        # Trailing logic using ATR of each leg
        if trailing_on:
            ce_atr = float(ce_df["ATR"].iloc[i]) if not math.isnan(ce_df["ATR"].iloc[i]) else None
            pe_atr = float(pe_df["ATR"].iloc[i]) if not math.isnan(pe_df["ATR"].iloc[i]) else None

            # Activate trailing once profit >= trail_start*ATR (for SHORT, profit when price drops)
            if ce_atr:
                unreal_ce = ce_leg.entry - ce_close
                if unreal_ce >= trail_start * ce_atr:
                    ce_leg.trail_active = True
                    new_sl = round(ce_close + trail_mult * ce_atr, 2)  # for short, SL above price
                    if new_sl < ce_leg.sl_price:
                        ce_leg.sl_price = new_sl
                        print(f"🔧 CE trail @ {bar_dt.time()} → SL {ce_leg.sl_price:.2f}")

            if pe_atr:
                unreal_pe = pe_leg.entry - pe_close
                if unreal_pe >= trail_start * pe_atr:
                    pe_leg.trail_active = True
                    new_sl = round(pe_close + trail_mult * pe_atr, 2)
                    if new_sl < pe_leg.sl_price:
                        pe_leg.sl_price = new_sl
                        print(f"🔧 PE trail @ {bar_dt.time()} → SL {pe_leg.sl_price:.2f}")

        # Check individual SL hits
        if not ce_sl_hit and ce_close >= ce_leg.sl_price:
            ce_sl_hit = True
            msg = f"⛔ CE SL hit @ {bar_dt.time()} | close {ce_close:.2f} ≥ SL {ce_leg.sl_price:.2f}"
            print(msg); send_telegram(msg)

        if not pe_sl_hit and pe_close >= pe_leg.sl_price:
            pe_sl_hit = True
            msg = f"⛔ PE SL hit @ {bar_dt.time()} | close {pe_close:.2f} ≥ SL {pe_leg.sl_price:.2f}"
            print(msg); send_telegram(msg)

        # Combo target check (profit when combined premium falls)
        combo = ce_close + pe_close
        if not combo_hit and combo <= combo_tgt_val:
            combo_hit = True
            msg = f"🎯 Combo target hit @ {bar_dt.time()} | {combo:.2f} ≤ {combo_tgt_val:.2f}"
            print(msg); send_telegram(msg)

        pnl_rows.append(
            {"time": bar_dt.isoformat(timespec="minutes"),
             "ce": ce_close, "pe": pe_close,
             "ce_sl": ce_leg.sl_price, "pe_sl": pe_leg.sl_price,
             "combo": combo}
        )

        if (ce_sl_hit and pe_sl_hit) or combo_hit:
            break

    # Exit prices (last processed bar or squareoff)
    if pnl_rows:
        last = pnl_rows[-1]
        ce_exit = last["ce"]
        pe_exit = last["pe"]
        exit_t = last["time"]
    else:
        # no bars processed after entry, fallback to last close
        ce_exit = float(ce_df["Close"].iloc[-1])
        pe_exit = float(pe_df["Close"].iloc[-1])
        exit_t = ce_df["date"].iloc[-1].isoformat(timespec="minutes")

    # P&L (short premium)
    ce_pnl = (ce_leg.entry - ce_exit) * lot_size * lots
    pe_pnl = (pe_leg.entry - pe_exit) * lot_size * lots
    total_pnl = ce_pnl + pe_pnl

    summary = (
        f"📦 Exit @ {exit_t}\n"
        f"• CE: entry {ce_leg.entry:.2f} → exit {ce_exit:.2f} | P&L {ce_pnl:.0f}\n"
        f"• PE: entry {pe_leg.entry:.2f} → exit {pe_exit:.2f} | P&L {pe_pnl:.0f}\n"
        f"• <b>Total P&L</b>: {total_pnl:.0f}"
    )
    print(summary)
    send_telegram(summary)

    # Write reports
    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump({
            "when": tz_now_ist().isoformat(timespec="seconds"),
            "underlying": underlying,
            "atm": atm,
            "expiry": str(expiry_d),
            "entry_time": entry_time,
            "squareoff": squareoff,
            "ce_entry": ce_leg.entry, "pe_entry": pe_leg.entry,
            "ce_sl": ce_leg.sl_price, "pe_sl": pe_leg.sl_price,
            "combo_target": combo_tgt_val,
            "exit_time": exit_t,
            "ce_exit": ce_exit, "pe_exit": pe_exit,
            "pnl": {"ce": ce_pnl, "pe": pe_pnl, "total": total_pnl},
            "trail": {
                "enabled": trailing_on, "type": trail_type,
                "start_atr": trail_start, "atr_mult": trail_mult
            },
            "bars": pnl_rows
        }, f, indent=2)

    print(f"📁 Wrote {os.path.join(out_dir, 'latest.json')}")


# ------------------- CLI -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out_dir", default="./reports")
    args = parser.parse_args()

    try:
        run_intraday(args.config, args.out_dir)
    except Exception as e:
        print(f"❌ Runner failed: {e}")
        sys.exit(1)
