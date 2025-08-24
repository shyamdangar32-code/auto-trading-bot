# runner_intraday_options.py
#
# Intraday ATM options (BANKNIFTY) — paper-only dual-momentum:
# - Decide ATM strike from BANKNIFTY spot
# - Pick nearest-weekly CE & PE
# - 5m candles, EMA(9/21) crossover
# - One side at a time; SL/Target; hard square-off time
#
# Needs env: ZERODHA_API_KEY + ZERODHA_ACCESS_TOKEN (+ Telegram env for alerts)

import os
from datetime import datetime, timedelta, time as dtime
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from bot.config import get_cfg
from bot.utils import ensure_dir, send_telegram

try:
    from kiteconnect import KiteConnect
except Exception:
    raise SystemExit("kiteconnect is required. Install with: pip install kiteconnect")

IST = "Asia/Kolkata"

# ---------- time helpers ----------
def _now_ist() -> datetime:
    return pd.Timestamp.now(tz=IST).to_pydatetime()

def _today_ist() -> datetime:
    n = _now_ist()
    return n.replace(hour=0, minute=0, second=0, microsecond=0)

def _ist_from_hhmm(hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    base = _today_ist()
    return base.replace(hour=h, minute=m)

def pct(a, b) -> float:
    return (b - a) / a * 100.0 if a else 0.0

# ---------- Zerodha ----------
def get_kite() -> KiteConnect:
    api_key = os.getenv("ZERODHA_API_KEY", "").strip()
    access  = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
    if not api_key or not access:
        raise SystemExit("ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN missing")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    kite.profile()  # sanity check
    return kite

def banknifty_spot_ltp(kite: KiteConnect) -> float:
    # Index quote key format:
    q = kite.quote(["NSE:NIFTY BANK"])
    return float(q["NSE:NIFTY BANK"]["last_price"])

def instruments_df(kite: KiteConnect) -> pd.DataFrame:
    ins = kite.instruments("NFO")
    df = pd.DataFrame(ins)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["strike"] = pd.to_numeric(df["strike"])
    return df

def pick_weekly_atm(df: pd.DataFrame, underlying: str, atm: int) -> Tuple[Dict, Dict]:
    # prefer nearest-expiry rows for exact strike; fallback to nearest strike
    rows = df[(df["name"] == underlying) & (df["segment"] == "NFO-OPT") & (df["strike"] == atm)]
    if rows.empty:
        near = df[(df["name"] == underlying) & (df["segment"] == "NFO-OPT")]
        near = near.iloc[(near["strike"] - atm).abs().argsort()[:20]]
        near = near.sort_values(["expiry", "strike"])
        ce = near[near["instrument_type"] == "CE"].iloc[0].to_dict()
        pe = near[near["instrument_type"] == "PE"].iloc[0].to_dict()
        return ce, pe
    expiry = rows.sort_values("expiry").iloc[0]["expiry"]
    ce = rows[(rows["instrument_type"] == "CE") & (rows["expiry"] == expiry)].iloc[0].to_dict()
    pe = rows[(rows["instrument_type"] == "PE") & (rows["expiry"] == expiry)].iloc[0].to_dict()
    return ce, pe

def hist_df(kite: KiteConnect, token: int, start: datetime, end: datetime, tf: str) -> pd.DataFrame:
    candles = kite.historical_data(instrument_token=token, from_date=start, to_date=end, interval=tf)
    d = pd.DataFrame(candles)
    if d.empty:
        return d
    d.rename(columns={"date":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
    d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize("UTC").dt.tz_convert(IST)
    return d.sort_values("Date").reset_index(drop=True)

# ---------- simple indicators ----------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def add_signals(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    d = df.copy()
    d["ema_fast"] = ema(d["Close"], fast)
    d["ema_slow"] = ema(d["Close"], slow)
    d["up"]   = (d["ema_fast"] > d["ema_slow"]).astype(int)
    d["xup"]  = (d["up"].diff() == 1)
    d["xdown"]= (d["up"].diff() == -1)
    return d

# ---------- simulator ----------
def simulate_day(ce: pd.DataFrame, pe: pd.DataFrame, opt: dict) -> dict:
    estart = _ist_from_hhmm(opt["entry_start"])
    eend   = _ist_from_hhmm(opt["entry_end"])
    hard   = _ist_from_hhmm(opt["hard_exit"])
    sl_p   = float(opt["sl_pct"])
    tp_p   = float(opt["target_pct"])
    qty    = int(opt["qty"])

    ce = add_signals(ce, opt["ema_fast"], opt["ema_slow"])
    pe = add_signals(pe, opt["ema_fast"], opt["ema_slow"])

    # union timeline (5m)
    all_ts = pd.date_range(
        start=min(ce["Date"].min(), pe["Date"].min()),
        end=max(ce["Date"].max(), pe["Date"].max()),
        freq="5min", tz=IST
    )
    ce2 = ce.set_index("Date").reindex(all_ts).ffill().reset_index().rename(columns={"index":"Date"})
    pe2 = pe.set_index("Date").reindex(all_ts).ffill().reset_index().rename(columns={"index":"Date"})

    trades = []
    pos = None  # {'leg': 'CE'|'PE', 'entry_px': float, 'entry_time': dt, 'qty': int}

    for i, ts in enumerate(all_ts):
        if ts > hard:
            if pos is not None:
                last_px = ce2.loc[i, "Close"] if pos["leg"] == "CE" else pe2.loc[i, "Close"]
                trades.append({"leg":pos["leg"], "entry_time":pos["entry_time"], "entry_px":pos["entry_px"],
                               "exit_time":ts, "exit_px":last_px, "qty":pos["qty"], "exit_reason":"HARD_EXIT"})
                pos = None
            break

        ce_r, pe_r = ce2.loc[i], pe2.loc[i]

        if pos is not None:
            last_px = ce_r["Close"] if pos["leg"] == "CE" else pe_r["Close"]
            chg = pct(pos["entry_px"], last_px)
            if chg <= -sl_p:
                trades.append({**pos, "exit_time":ts, "exit_px":last_px, "exit_reason":"SL"})
                pos = None
            elif chg >= tp_p:
                trades.append({**pos, "exit_time":ts, "exit_px":last_px, "exit_reason":"TP"})
                pos = None
            else:
                if pos["leg"] == "CE" and bool(ce_r["xdown"]):
                    trades.append({**pos, "exit_time":ts, "exit_px":last_px, "exit_reason":"XDOWN"})
                    pos = None
                elif pos["leg"] == "PE" and bool(pe_r["xdown"]):
                    trades.append({**pos, "exit_time":ts, "exit_px":last_px, "exit_reason":"XDOWN"})
                    pos = None

        if pos is None and (estart <= ts <= eend):
            if bool(ce_r["xup"]) and not bool(pe_r["xup"]):
                pos = {"leg":"CE", "entry_time":ts, "entry_px":ce_r["Close"], "qty":qty}
            elif bool(pe_r["xup"]) and not bool(ce_r["xup"]):
                pos = {"leg":"PE", "entry_time":ts, "entry_px":pe_r["Close"], "qty":qty}

    if pos is not None:
        ts = ce2.iloc[-1]["Date"]
        last_px = ce2.iloc[-1]["Close"] if pos["leg"] == "CE" else pe2.iloc[-1]["Close"]
        trades.append({**pos, "exit_time":ts, "exit_px":last_px, "exit_reason":"EOD_CLOSE"})
        pos = None

    if len(trades) == 0:
        return {"trades": pd.DataFrame(), "summary": {"n_trades": 0, "win_rate": 0.0, "total_PnL": 0.0}}

    df_tr = pd.DataFrame(trades)
    # simple PnL per lot (premium move * qty)
    df_tr["PnL"] = (df_tr["exit_px"] - df_tr["entry_px"]) * df_tr["qty"]
    win_rate = float((df_tr["PnL"] > 0).mean() * 100.0)
    total = float(df_tr["PnL"].sum())
    return {"trades": df_tr, "summary": {"n_trades": int(len(df_tr)), "win_rate": win_rate, "total_PnL": total}}

# ---------- main ----------
def main():
    cfg = get_cfg()
    opt = cfg.get("options_intraday", {})
    if not opt.get("enabled", False):
        print("options_intraday.enabled = false; nothing to do.")
        return

    kite = get_kite()
    print("✅ Zerodha token OK.")

    # window for warm-up
    today = _today_ist()
    start = today - timedelta(days=2)
    end   = today + timedelta(days=1)

    # ATM selection
    spot = banknifty_spot_ltp(kite)
    step = int(opt.get("strike_step", 100))
    atm  = int(round(spot / step) * step)
    print(f"ℹ️ BANKNIFTY spot {spot:.1f} → ATM {atm}")

    # instruments
    ins = instruments_df(kite)
    ce_i, pe_i = pick_weekly_atm(ins, opt.get("underlying", "BANKNIFTY"), atm)
    ce_tok, pe_tok = int(ce_i["instrument_token"]), int(pe_i["instrument_token"])
    print(f"CE {ce_i['tradingsymbol']} (expiry {str(ce_i['expiry'].date())})  token={ce_tok}")
    print(f"PE {pe_i['tradingsymbol']} (expiry {str(pe_i['expiry'].date())})  token={pe_tok}")

    tf_map = {"5m": "5minute", "15m": "15minute"}
    tf = tf_map.get(opt.get("interval","5m"), "5minute")
    ce = hist_df(kite, ce_tok, start, end, tf)
    pe = hist_df(kite, pe_tok, start, end, tf)
    if ce.empty or pe.empty:
        raise SystemExit("No candles for CE/PE (holiday or token issue).")

    # keep only today's IST bars
    ce = ce[ce["Date"].dt.tz_convert(IST).dt.date == today.date()].reset_index(drop=True)
    pe = pe[pe["Date"].dt.tz_convert(IST).dt.date == today.date()].reset_index(drop=True)
    print(f"📈 Candles today — CE: {len(ce)}  PE: {len(pe)}")

    results = simulate_day(ce, pe, opt)
    summ = results["summary"]
    print("📊 Summary:", summ)

    # Telegram
    msg = (
        "🤖 Intraday Options (paper)\n"
        f"Underlying: BANKNIFTY | ATM: {atm}\n"
        f"EMA({opt['ema_fast']}/{opt['ema_slow']}), SL {opt['sl_pct']}%, TP {opt['target_pct']}%\n"
        f"Trades: {summ['n_trades']} | Win rate: {summ['win_rate']:.1f}% | PnL(sum): {summ['total_PnL']:.2f}"
    )
    send_telegram(msg)

    # Save trades
    out_dir = cfg.get("out_dir", "reports")
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "intraday_options_trades.csv")
    results["trades"].to_csv(path, index=False)
    print(f"💾 Saved: {path}")

if __name__ == "__main__":
    main()
