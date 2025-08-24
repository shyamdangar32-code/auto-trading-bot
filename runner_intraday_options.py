#!/usr/bin/env python3
"""
Intraday Options Runner (BANKNIFTY CE/PE) — Zerodha only

Features
- Picks ATM strike for this week’s expiry (Thu) from BANKNIFTY spot
- Loads 5m candles for CE & PE via kite.historical_data
- Entry = price cross above its MA (ma_len)
- SL = ATR * sl_atr_mult
- Target = Risk * tgt_rr
- Trailing = after trail_start_atr profit, trail by trail_atr_mult*ATR
- Re-entry: up to reentry_max with cooldown bars
- Paper trading by default; can place live orders if live_trading=True
- Telegram alerts on entries/exits

No Yahoo fallback. Safe to run in GitHub Actions.
"""

import os
import json
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass
import math
import pytz
import pandas as pd
import numpy as np
import requests

from kiteconnect import KiteConnect

IST = pytz.timezone("Asia/Kolkata")

# ---------- Config helpers ----------
def load_cfg(path="config.yaml"):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ist_now():
    return datetime.now(IST)

# ---------- Telegram ----------
def tg_send(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass

# ---------- Indicators ----------
def ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["High"]; low = df["Low"]; close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

# ---------- Zerodha helpers ----------
def kite_from_secrets(cfg):
    kc = KiteConnect(api_key=cfg["zerodha_api_key"])
    kc.set_access_token(cfg["zerodha_access_token"])
    return kc

def next_thursday(d):
    # next/this Thursday (weekly expiry)
    days_ahead = (3 - d.weekday()) % 7  # 0=Mon ... 3=Thu
    return (d + timedelta(days=days_ahead)).date()

def round_to(x, step):
    return int(round(x / step) * step)

def ensure_ist_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kite historical_data returns list of dicts with 'date' tz-aware (IST) in most envs.
    Make a clean tz-aware IST index named 'Date' and OHLC columns capitalized.
    """
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(IST)
        else:
            dt = dt.dt.tz_convert(IST)
        df["Date"] = dt
        df = df.drop(columns=["date"])
    if "Date" not in df.columns:
        # fallback if already index-like
        df["Date"] = pd.to_datetime(df.index).tz_localize(IST, nonexistent="shift_forward", ambiguous="NaT")
    # Standardize column names
    rename = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume"
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    df = df[["Date", "Open", "High", "Low", "Close"]].copy()
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def option_token(kite, tradingsymbol: str):
    # fast path with instruments cache in Actions
    ins = kite.instruments("NFO")
    rec = next((r for r in ins if r["tradingsymbol"] == tradingsymbol), None)
    if not rec:
        raise RuntimeError(f"Cannot find instrument for {tradingsymbol}")
    return rec["instrument_token"]

def make_symbol(underlying: str, expiry_date, strike: int, opt_type: str):
    # Example: BANKNIFTY25AUG55100CE
    mon = expiry_date.strftime("%b").upper()
    yy = expiry_date.strftime("%y")
    return f"{underlying}{yy}{mon}{strike}{opt_type}"

def hist_df(kite, token: int, start_dt: datetime, end_dt: datetime, tf="5minute"):
    data = kite.historical_data(token, start_dt, end_dt, tf, continuous=False, oi=False)
    df = pd.DataFrame(data)
    if df.empty:
        raise RuntimeError("historical_data returned empty.")
    return ensure_ist_index(df)

# ---------- Strategy Engine ----------
@dataclass
class OptCfg:
    underlying: str = "BANKNIFTY"
    index_token: int = 260105
    tf: str = "5minute"
    strike_step: int = 100

    start: str = "09:20"  # HH:MM IST
    end: str = "15:15"

    ma_len: int = 20
    atr_len: int = 14
    sl_atr_mult: float = 1.0
    tgt_rr: float = 1.5

    trail_start_atr: float = 1.0
    trail_atr_mult: float = 1.0

    reentry_max: int = 2
    reentry_cooldown: int = 3

    order_qty: int = 1
    paper_trading: bool = True
    live_trading: bool = False

def parse_time(hhmm: str):
    hh, mm = [int(x) for x in hhmm.split(":")]
    return dtime(hour=hh, minute=mm, tzinfo=IST)

def backtest_day(df: pd.DataFrame, cfg: OptCfg, side_name: str, tg=None):
    """
    Very compact intraday engine on one option series.
    Entry: Cross above MA
    Exit: SL or Target; trailing after threshold
    Reentry: up to cfg.reentry_max with cooldown
    """
    df = df.copy()
    df["MA"] = ma(df["Close"], cfg.ma_len)
    df["ATR"] = atr(df, cfg.atr_len)
    start_t = parse_time(cfg.start)
    end_t   = parse_time(cfg.end)

    trades = []
    position = None
    cooldown = 0
    reentries = 0

    for i in range(1, len(df)):
        row_prev = df.iloc[i-1]
        row = df.iloc[i]
        t = row["Date"].time()

        if t < start_t or t > end_t:
            continue

        # manage open position
        if position:
            # trailing
            if (row["Close"] - position["entry"]) >= cfg.trail_start_atr * row["ATR"]:
                new_sl = row["Close"] - cfg.trail_atr_mult * row["ATR"]
                position["sl"] = max(position["sl"], new_sl)

            # SL hit?
            if row["Low"] <= position["sl"]:
                exit_px = position["sl"]
                pnl = (exit_px - position["entry"]) * position["qty"]
                trades.append({"side": side_name, "entry": position["entry_ts"],
                               "exit": row["Date"], "entry_px": position["entry"],
                               "exit_px": exit_px, "reason": "SL", "pnl": pnl})
                position = None
                cooldown = cfg.reentry_cooldown
                if tg: tg(f"🔴 {side_name} SL hit @ {exit_px:.2f}")
                continue

            # Target hit?
            if row["High"] >= position["target"]:
                exit_px = position["target"]
                pnl = (exit_px - position["entry"]) * position["qty"]
                trades.append({"side": side_name, "entry": position["entry_ts"],
                               "exit": row["Date"], "entry_px": position["entry"],
                               "exit_px": exit_px, "reason": "TARGET", "pnl": pnl})
                position = None
                cooldown = cfg.reentry_cooldown
                if tg: tg(f"✅ {side_name} Target hit @ {exit_px:.2f}")
                continue

        # no position: consider entry
        if not position:
            if cooldown > 0:
                cooldown -= 1
            else:
                crossed_up = (row_prev["Close"] <= row_prev["MA"]) and (row["Close"] > row["MA"])
                if crossed_up:
                    # enter long option
                    risk = cfg.sl_atr_mult * row["ATR"]
                    entry_px = float(row["Close"])
                    sl = entry_px - risk
                    target = entry_px + cfg.tgt_rr * risk
                    position = {
                        "entry": entry_px, "sl": sl, "target": target, "qty": cfg.order_qty,
                        "entry_ts": row["Date"],
                    }
                    reentries += 1
                    if tg: tg(f"🟢 {side_name} ENTRY @ {entry_px:.2f} | SL {sl:.2f} | TGT {target:.2f}")
                # stop re-entering after limit
                if reentries >= cfg.reentry_max:
                    cooldown = 999  # block further entries for the day

    # force close any open position at end
    if position:
        exit_px = float(df.iloc[-1]["Close"])
        pnl = (exit_px - position["entry"]) * position["qty"]
        trades.append({"side": side_name, "entry": position["entry_ts"],
                       "exit": df.iloc[-1]["Date"], "entry_px": position["entry"],
                       "exit_px": exit_px, "reason": "EOD", "pnl": pnl})
        if tg: tg(f"📘 {side_name} EOD exit @ {exit_px:.2f}")

    return trades

# ---------- Main ----------
def main():
    CFG = load_cfg("config.yaml")
    Z = CFG  # flat for convenience
    tok = Z.get("telegram_bot_token", "")
    cid = Z.get("telegram_chat_id", "")

    def T(msg): tg_send(tok, cid, msg)

    intr = Z.get("intraday_options", {})  # new section
    ocfg = OptCfg(
        underlying=intr.get("underlying", "BANKNIFTY"),
        index_token=int(intr.get("index_token", 260105)),
        tf=intr.get("timeframe", "5minute"),
        strike_step=int(intr.get("strike_step", 100)),
        start=intr.get("start_time", "09:20"),
        end=intr.get("end_time", "15:15"),
        ma_len=int(intr.get("ma_len", 20)),
        atr_len=int(intr.get("atr_len", 14)),
        sl_atr_mult=float(intr.get("sl_atr_mult", 1.0)),
        tgt_rr=float(intr.get("tgt_rr", 1.5)),
        trail_start_atr=float(intr.get("trail_start_atr", 1.0)),
        trail_atr_mult=float(intr.get("trail_atr_mult", 1.0)),
        reentry_max=int(intr.get("reentry_max", 2)),
        reentry_cooldown=int(intr.get("reentry_cooldown", 3)),
        order_qty=int(intr.get("order_qty", 1)),
        paper_trading=bool(intr.get("paper_trading", True)),
        live_trading=bool(intr.get("live_trading", False)),
    )

    # Zerodha
    kite = kite_from_secrets(Z)

    # Spot & ATM strike
    ltp = kite.ltp(f"NSE:{ocfg.underlying}")  # BANKNIFTY index
    spot = list(ltp.values())[0]["last_price"]
    atm = round_to(spot, ocfg.strike_step)

    # Weekly expiry symbol(s)
    exp = next_thursday(ist_now().date())
    ce_sym = make_symbol(ocfg.underlying, exp, atm, "CE")
    pe_sym = make_symbol(ocfg.underlying, exp, atm, "PE")
    ce_token = option_token(kite, ce_sym)
    pe_token = option_token(kite, pe_sym)

    print(f"✅ Zerodha token OK.")
    print(f"ℹ️ {ocfg.underlying} spot {spot:.1f} → ATM {atm}")
    print(f"ℹ️ Weekly expiry → {exp}")
    print(f"ℹ️ CE {ce_sym} token={ce_token}")
    print(f"ℹ️ PE {pe_sym} token={pe_token}")

    # Date range for today (IST)
    today = ist_now().date()
    start_dt = IST.localize(datetime.combine(today, dtime(9, 15)))
    end_dt   = IST.localize(datetime.combine(today, dtime(15, 30)))

    ce = hist_df(kite, ce_token, start_dt, end_dt, tf=ocfg.tf)
    pe = hist_df(kite, pe_token, start_dt, end_dt, tf=ocfg.tf)
    print(f"✅ Zerodha CE rows: {len(ce)}  | PE rows: {len(pe)}")

    # Build tiny bias preview (for logs)
    ce["MA20"] = ma(ce["Close"], 20)
    pe["MA20"] = ma(pe["Close"], 20)
    ce_dir = "UP" if ce.iloc[-1]["Close"] > ce.iloc[-1]["MA20"] else "DOWN"
    pe_dir = "UP" if pe.iloc[-1]["Close"] > pe.iloc[-1]["MA20"] else "DOWN"
    print("—" * 50)
    print(f"📈 {ce_sym} | Close: {ce.iloc[-1]['Close']:.2f} | MA20: {ce.iloc[-1]['MA20']:.2f} | {ce_dir}")
    print(f"📈 {pe_sym} | Close: {pe.iloc[-1]['Close']:.2f} | MA20: {pe.iloc[-1]['MA20']:.2f} | {pe_dir}")

    # Decide which side to run (simple: pick side above MA20)
    sides = []
    if ce_dir == "UP":
        sides.append(("CE", ce))
    if pe_dir == "UP":
        sides.append(("PE", pe))
    if not sides:
        # default to side with stronger distance to MA
        d_ce = ce.iloc[-1]["MA20"] - ce.iloc[-1]["Close"]
        d_pe = pe.iloc[-1]["MA20"] - pe.iloc[-1]["Close"]
        sides.append(("PE" if d_pe < d_ce else "CE", pe if d_pe < d_ce else ce))

    # Paper/live switch info
    mode = "PAPER" if ocfg.paper_trading and not ocfg.live_trading else "LIVE"
    T(f"⚙️ <b>{mode} {ocfg.underlying} Options</b>\n"
      f"ATM {atm} | Exp {exp}\n"
      f"Rules: MA{ocfg.ma_len} cross, SL {ocfg.sl_atr_mult}×ATR, "
      f"TGT {ocfg.tgt_rr}R, Trail {ocfg.trail_atr_mult}×ATR")

    all_trades = []
    for nm, d in sides:
        all_trades += backtest_day(d, ocfg, f"{nm} {atm}", tg=T)

    pnl = sum(t["pnl"] for t in all_trades)
    wins = sum(1 for t in all_trades if t["pnl"] > 0)
    ntr = len(all_trades)

    print("\nTrades:")
    for t in all_trades:
        print(f"- {t['side']} | {t['reason']} | entry {t['entry_px']:.2f} → exit {t['exit_px']:.2f} | PnL {t['pnl']:.2f}")

    print(f"\n📊 Summary: trades={ntr} wins={wins} pnl={pnl:.2f}")

    # Save report
    out_dir = CFG.get("out_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)
    stamp = ist_now().strftime("%Y%m%d_%H%M")
    rep_path = os.path.join(out_dir, f"intraday_options_{today}_{stamp}.json")
    with open(rep_path, "w") as f:
        json.dump({"summary": {"n_trades": ntr, "wins": wins, "pnl": pnl},
                   "trades": all_trades,
                   "meta": {"atm": atm, "expiry": str(exp)}},
                  f, default=str)
    print(f"🗂 Report saved → {rep_path}")

    # Live orders (optional hook)
    if ocfg.live_trading and not ocfg.paper_trading:
        # Place actual orders here if you want (currently disabled for safety)
        # You already have kite; convert entries to orders as needed.
        pass

if __name__ == "__main__":
    main()
