#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intraday ATM short-straddle (paper) with REAL signals printed + optional Telegram alerts.
- Uses Zerodha-only data (no Yahoo fallback)
- CE+PE at-the-money selection for weekly expiry
- Entry time, SL%, combined target%, ATR trailing, re-entries, cooldown, square-off
- Logs all events; writes reports/latest.json

Will NOT place any orders. (send_only_signals/paper/live flags remain unchanged)
"""

import os
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, time, timedelta, timezone

import pandas as pd

from bot.config import load_config
from bot.data_io import _get_kite  # already in your repo; validates token & returns KiteConnect


# ---------------------------- helpers ----------------------------

IST = timezone(timedelta(hours=5, minutes=30))

def ist_today_str():
    return datetime.now(IST).strftime("%Y-%m-%d")

def to_ist(dt):
    return dt.astimezone(IST) if dt.tzinfo else dt.replace(tzinfo=IST)

def parse_hhmm(s: str) -> time:
    hh, mm = s.strip().split(":")
    return time(int(hh), int(mm), tzinfo=IST)

def nearest_atm_strike(spot, step):
    return int(round(spot / step) * step)

def tr_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """True Range + ATR on OHLC."""
    prev_close = df["Close"].shift(1)
    ranges = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1)
    tr = ranges.max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def send_telegram(text: str):
    """
    Sends a Telegram message if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID envs are present
    or if the same are set in config.yaml under intraday_options.
    """
    import requests

    # Prefer env (from secrets). If missing, try config fields.
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    # Fallback to config.yaml if envs are empty
    if not token or not chat_id:
        cfg = load_config()
        tcfg = cfg.get("intraday_options", {})
        token = token or (tcfg.get("telegram_bot_token") or "").strip()
        chat_id = chat_id or (tcfg.get("telegram_chat_id") or "").strip()

    if not token or not chat_id:
        return  # silently skip

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text}
        requests.post(url, json=payload, timeout=7)
    except Exception:
        # don't crash the run if Telegram has an issue
        pass


# ---------------------------- strategy structs ----------------------------

@dataclass
class Leg:
    symbol: str
    token: int
    entry: float = math.nan
    stop: float = math.nan
    trail_active: bool = False
    exit: float = math.nan
    status: str = "PENDING"  # PENDING / ACTIVE / EXIT

@dataclass
class TradeEvent:
    ts: str
    kind: str
    msg: str


# ---------------------------- core runner ----------------------------

def main():
    cfg = load_config()
    tz = cfg.get("tz", "Asia/Kolkata")

    icfg = cfg.get("intraday_options", {})
    underlying = icfg.get("underlying", "BANKNIFTY").upper()
    lots = int(icfg.get("lots", 1))
    lot_size = int(icfg.get("lot_size", 15))
    interval = icfg.get("interval", "5minute")

    entry_t = parse_hhmm(icfg.get("entry_time", "09:30"))
    squareoff_t = parse_hhmm(icfg.get("squareoff", "15:10"))
    sim_date = (icfg.get("date") or "").strip() or ist_today_str()

    # Risk & trailing
    leg_sl_pct = float(icfg.get("leg_sl_percent", 30.0))
    tgt_pct = float(icfg.get("combined_target_percent", 50.0))
    trailing_enabled = bool(icfg.get("trailing_enabled", True))
    trail_type = icfg.get("trail_type", "atr")
    trail_start_atr = float(icfg.get("trail_start_atr", 1.0))
    trail_atr_mult = float(icfg.get("trail_atr_mult", 1.5))
    reentry_max = int(icfg.get("reentry_max", 0))
    reentry_cooldown = int(icfg.get("reentry_cooldown", 2))

    print(f"✅ Zerodha token OK.")
    kite = _get_kite()  # prints profile ok in data_io._get_kite

    # ------------- spot, strike step & expiry selection -------------
    # Spot via quote for index
    index_map = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK"}
    q = kite.ltp([index_map[underlying]])
    spot = list(q.values())[0]["last_price"]
    step = 50 if underlying == "NIFTY" else 100
    atm = nearest_atm_strike(spot, step)

    print(f"ℹ️  {underlying} spot {spot:.1f} → ATM {atm}")
    # Weekly expiry for given sim_date (Thurs for NSE)
    # Find next Thursday >= sim_date
    d = datetime.strptime(sim_date, "%Y-%m-%d").replace(tzinfo=IST)
    while d.weekday() != 3:  # 0=Mon ... 3=Thu
        d += timedelta(days=1)
    expiry_str = d.strftime("%Y-%m-%d")
    print(f"ℹ️  Weekly expiry → {expiry_str}")

    # ------------- instrument lookup for CE/PE -------------
    # Use instruments() once, filter locally for performance.
    instruments = kite.instruments("NFO")
    ce_token = pe_token = None
    ce_tradingsym = pe_tradingsym = None

    suffix = "CE", "PE"
    for ins in instruments:
        try:
            if ins.get("segment") != "NFO-OPT":
                continue
            if underlying not in ins.get("name", ""):
                continue
            if int(ins.get("strike", 0)) != atm:
                continue
            # normalize expiry compare by date only
            ins_exp = pd.to_datetime(ins["expiry"]).strftime("%Y-%m-%d")
            if ins_exp != expiry_str:
                continue
            tsym = ins["tradingsymbol"]
            tok = int(ins["instrument_token"])
            if tsym.endswith("CE"):
                ce_token, ce_tradingsym = tok, tsym
            elif tsym.endswith("PE"):
                pe_token, pe_tradingsym = tok, tsym
        except Exception:
            continue

    if not ce_token or not pe_token:
        raise RuntimeError(f"Could not find CE/PE tokens for {underlying} {expiry_str} @ {atm}")

    print(f"✅ Using expiry {expiry_str} | CE token: {ce_token} | PE token: {pe_token}")

    # ------------- fetch OHLC for options for the sim day -------------
    def hist(tok):
        # We need only the sim_date data; use from->to same day + 1
        start = pd.Timestamp(sim_date + " 09:15:00", tz=IST)
        end   = pd.Timestamp(sim_date + " 15:30:00", tz=IST)
        candles = kite.historical_data(
            instrument_token=int(tok),
            from_date=start.to_pydatetime(),
            to_date=end.to_pydatetime(),
            interval=interval,
        )
        df = pd.DataFrame(candles)
        df.rename(columns={"date": "Date", "open": "Open", "high": "High",
                           "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_convert(IST)
        return df

    ce_df = hist(ce_token)
    pe_df = hist(pe_token)

    print(f"📈 {ce_tradingsym} | rows: {len(ce_df)} | Close: {ce_df['Close'].iloc[-1]}")
    print(f"📉 {pe_tradingsym} | rows: {len(pe_df)} | Close: {pe_df['Close'].iloc[-1]}")

    # ------------- prepare ATR (for trailing) -------------
    ce_df["ATR"] = tr_atr(ce_df, 14)
    pe_df["ATR"] = tr_atr(pe_df, 14)

    # ------------- simulate from entry_time to squareoff -------------
    events: list[TradeEvent] = []
    def log(kind, msg, ts=None):
        ts = ts or datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        print(msg)
        events.append(TradeEvent(ts=ts, kind=kind, msg=msg))

    # find entry bar index (first bar whose time >= entry_time)
    def find_entry_idx(df: pd.DataFrame) -> int:
        for i, dt in enumerate(df["Date"]):
            if dt.timetz() >= entry_t:
                return i
        return -1

    ent_idx = find_entry_idx(ce_df)
    if ent_idx < 0:
        log("WARN", "No entry bar found (check entry_time/interval). Exiting.")
        _write_report(cfg, underlying, expiry_str, atm, ce_tradingsym, pe_tradingsym, events)
        return

    # legs (we SELL both, so SL is above price)
    ce = Leg(symbol=ce_tradingsym, token=ce_token)
    pe = Leg(symbol=pe_tradingsym, token=pe_token)

    active_reentries = 0
    cooldown_left = 0
    combined_target = None

    # walk bars from entry index to end
    i = ent_idx
    while i < len(ce_df):
        ts = ce_df["Date"].iloc[i]
        if ts.timetz() >= squareoff_t:
            log("EXIT", f"⏰ Square-off at {ts.strftime('%H:%M')}. Exiting open legs.", ts.strftime("%Y-%m-%d %H:%M:%S"))
            if ce.status == "ACTIVE":
                ce.exit = ce_df["Close"].iloc[i]; ce.status = "EXIT"
            if pe.status == "ACTIVE":
                pe.exit = pe_df["Close"].iloc[i]; pe.status = "EXIT"
            break

        # Cooldown handling
        if cooldown_left > 0:
            cooldown_left -= 1

        # Entry
        if ce.status == "PENDING" and pe.status == "PENDING" and cooldown_left == 0:
            ce.entry = float(ce_df["Close"].iloc[i])
            pe.entry = float(pe_df["Close"].iloc[i])
            ce.stop  = ce.entry * (1 + leg_sl_pct / 100.0)
            pe.stop  = pe.entry * (1 + leg_sl_pct / 100.0)
            ce.status = pe.status = "ACTIVE"
            combined_entry = ce.entry + pe.entry
            combined_target = combined_entry * (1 - tgt_pct / 100.0) if tgt_pct > 0 else None
            msg = (f"🟢 ENTRY {ts.strftime('%H:%M')} | SELL {ce.symbol} @{ce.entry:.2f}, "
                   f"SELL {pe.symbol} @{pe.entry:.2f} | SLs: {ce.stop:.2f}/{pe.stop:.2f}")
            log("ENTRY", msg, ts.strftime("%Y-%m-%d %H:%M:%S"))
            send_telegram(f"[ENTRY] {underlying} {sim_date} {ts.strftime('%H:%M')}\n"
                          f"SELL {ce.symbol} @{ce.entry:.2f} SL {ce.stop:.2f}\n"
                          f"SELL {pe.symbol} @{pe.entry:.2f} SL {pe.stop:.2f}")

        # If active, check target/SL and trailing
        if ce.status == "ACTIVE" and pe.status == "ACTIVE":
            ce_px = float(ce_df["Close"].iloc[i])
            pe_px = float(pe_df["Close"].iloc[i])
            combined = ce_px + pe_px

            # Combined target
            if combined_target is not None and combined <= combined_target:
                ce.exit = ce_px; pe.exit = pe_px
                ce.status = pe.status = "EXIT"
                log("TARGET", f"🎯 TARGET hit {ts.strftime('%H:%M')} | Combined {combined:.2f} ≤ {combined_target:.2f}")
                send_telegram(f"[TARGET] {underlying} {ts.strftime('%H:%M')}\nCombined {combined:.2f} ≤ {combined_target:.2f}")
                # enable potential re-entry
                if active_reentries < reentry_max:
                    cooldown_left = reentry_cooldown
                    active_reentries += 1
                    ce = Leg(symbol=ce_tradingsym, token=ce_token)
                    pe = Leg(symbol=pe_tradingsym, token=pe_token)
                i += 1
                continue

            # SL checks (since we are short, SL is price rising above stop)
            if ce_px >= ce.stop or pe_px >= pe.stop:
                reason = []
                if ce_px >= ce.stop:
                    ce.exit = ce_px; ce.status = "EXIT"; reason.append(f"{ce.symbol} SL")
                if pe_px >= pe.stop:
                    pe.exit = pe_px; pe.status = "EXIT"; reason.append(f"{pe.symbol} SL")
                log("SL", f"❌ STOP hit {ts.strftime('%H:%M')} | " + " & ".join(reason))
                send_telegram(f"[STOP] {underlying} {ts.strftime('%H:%M')} | " + " & ".join(reason))
                # re-entry?
                if active_reentries < reentry_max:
                    cooldown_left = reentry_cooldown
                    active_reentries += 1
                    ce = Leg(symbol=ce_tradingsym, token=ce_token)
                    pe = Leg(symbol=pe_tradingsym, token=pe_token)
                i += 1
                continue

            # Trailing (simple ATR trailing for shorts)
            if trailing_enabled and trail_type == "atr":
                ce_atr = float(ce_df["ATR"].iloc[i])
                pe_atr = float(pe_df["ATR"].iloc[i])

                # Activate trailing once unrealized gain exceeds start*ATR (i.e., price moved our way)
                if not ce.trail_active:
                    if (ce.entry - ce_px) >= trail_start_atr * ce_atr:
                        ce.trail_active = True
                        log("TRAIL", f"🪢 {ce.symbol} trailing activated at {ts.strftime('%H:%M')}")
                if not pe.trail_active:
                    if (pe.entry - pe_px) >= trail_start_atr * pe_atr:
                        pe.trail_active = True
                        log("TRAIL", f"🪢 {pe.symbol} trailing activated at {ts.strftime('%H:%M')}")

                if ce.trail_active:
                    new_stop = ce_px + trail_atr_mult * ce_atr
                    if new_stop < ce.stop:
                        ce.stop = new_stop
                        log("TRAIL", f"🔧 {ce.symbol} stop trailed → {ce.stop:.2f}")
                if pe.trail_active:
                    new_stop = pe_px + trail_atr_mult * pe_atr
                    if new_stop < pe.stop:
                        pe.stop = new_stop
                        log("TRAIL", f"🔧 {pe.symbol} stop trailed → {pe.stop:.2f}")

        i += 1

    # final log if still pending (no entry bar met, etc.)
    if ce.status == "PENDING" and pe.status == "PENDING":
        log("INFO", "No entries taken for the day.")

    _write_report(cfg, underlying, expiry_str, atm, ce_tradingsym, pe_tradingsym, events)


def _write_report(cfg, underlying, expiry, atm, ce_sym, pe_sym, events):
    out_dir = cfg.get("out_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "ts": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "underlying": underlying,
        "expiry": expiry,
        "atm": atm,
        "ce": ce_sym,
        "pe": pe_sym,
        "events": [asdict(e) for e in events],
    }
    with open(os.path.join(out_dir, "latest.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("📦 Wrote reports/latest.json")


if __name__ == "__main__":
    main()
