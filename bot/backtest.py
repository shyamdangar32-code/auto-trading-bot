# bot/backtest.py
# Minimal, self-contained day-by-day backtester that uses the SAME
# risk settings from config.yaml and historical candles from Kite.
# It simulates ONE trade per day on ONE instrument token:
# enter at entry_time, exit on SL/Target/End-of-day.
#
# Writes:
#   reports/backtest/trades.csv
#   reports/backtest/equity_curve.csv
#   reports/backtest/summary.json

from __future__ import annotations

import os, json, argparse
from datetime import datetime, timedelta, date, time
from typing import List, Dict

import pandas as pd
from kiteconnect import KiteConnect

# ---- tiny config loader (works with your existing bot/config.py if present) ----
def _load_config():
    # Prefer your helper if it exists
    try:
        from bot.config import load_config  # type: ignore
        return load_config()
    except Exception:
        import yaml
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)

def ist_dt(d: date, hhmm: str) -> datetime:
    hh, mm = map(int, hhmm.split(":"))
    # naive datetime in IST wall-clock (Kite accepts naive local timestamps fine in GA)
    return datetime(d.year, d.month, d.day, hh, mm, 0)

def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def fetch_day_candles(kite: KiteConnect, token: int, d: date, interval: str) -> pd.DataFrame:
    # 09:15 -> 15:30 to be safe for all F&O segments
    start = ist_dt(d, "09:15")
    end   = ist_dt(d, "15:30")
    raw = kite.historical_data(token, start, end, interval=interval)
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    # ensure datetime and standard cols
    df["datetime"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"open":"o","high":"h","low":"l","close":"c","volume":"v"})
    df = df[["datetime","o","h","l","c","v"]].sort_values("datetime").reset_index(drop=True)
    return df

def simulate_day(
    df: pd.DataFrame,
    d: date,
    side: str,
    qty: int,
    entry_time: str,
    end_time: str,
    sl_pct: float,
    target_pct: float,
):
    """
    One trade per day:
    - Entry on/after entry_time at the first available candle close
    - Exit if SL/Target hit; otherwise exit at end_time close
    """
    if df.empty:
        return None  # holiday / no data

    # window
    et = ist_dt(d, entry_time)
    xt = ist_dt(d, end_time)

    df = df[(df["datetime"] >= et) & (df["datetime"] <= xt)].copy()
    if df.empty:
        return None

    # entry is the first candle's close at/after entry_time
    entry_row = df.iloc[0]
    entry_px = float(entry_row["c"])

    # compute absolute SL/Target
    if sl_pct and sl_pct > 0:
        sl_abs = entry_px * (1 - sl_pct) if side == "buy" else entry_px * (1 + sl_pct)
    else:
        sl_abs = None
    if target_pct and target_pct > 0:
        tgt_abs = entry_px * (1 + target_pct) if side == "buy" else entry_px * (1 - target_pct)
    else:
        tgt_abs = None

    exit_reason = "EOD"
    exit_px = float(df.iloc[-1]["c"])
    exit_time = df.iloc[-1]["datetime"]

    # scan candles to see which hits first
    for _, r in df.iterrows():
        high = float(r["h"])
        low  = float(r["l"])
        # buy: SL if low<=sl_abs, target if high>=tgt_abs
        # sell: SL if high>=sl_abs, target if low<=tgt_abs
        if sl_abs is not None:
            if side == "buy" and low <= sl_abs:
                exit_px = sl_abs
                exit_time = r["datetime"]
                exit_reason = "SL"
                break
            if side == "sell" and high >= sl_abs:
                exit_px = sl_abs
                exit_time = r["datetime"]
                exit_reason = "SL"
                break
        if tgt_abs is not None:
            if side == "buy" and high >= tgt_abs:
                exit_px = tgt_abs
                exit_time = r["datetime"]
                exit_reason = "TARGET"
                break
            if side == "sell" and low <= tgt_abs:
                exit_px = tgt_abs
                exit_time = r["datetime"]
                exit_reason = "TARGET"
                break

    # PnL
    if side == "buy":
        pnl = (exit_px - entry_px) * qty
        rr  = None if sl_pct in (None,0) else pnl / (entry_px * sl_pct * qty)
    else:
        pnl = (entry_px - exit_px) * qty
        rr  = None if sl_pct in (None,0) else pnl / (entry_px * sl_pct * qty)

    return {
        "date": d.isoformat(),
        "entry_time": df.iloc[0]["datetime"].isoformat(),
        "exit_time": exit_time.isoformat(),
        "side": side,
        "entry": round(entry_px, 2),
        "exit": round(exit_px, 2),
        "reason": exit_reason,
        "qty": qty,
        "pnl": round(pnl, 2),
        "rr": None if rr is None else round(rr, 3),
    }

def main():
    cfg = _load_config()
    trade = cfg.get("trade", {})
    entry_time  = trade.get("entry_time", "09:20")
    end_time    = trade.get("end_time",   "15:05")
    sl_pct      = float(trade.get("sl_pct", 0.25))
    target_pct  = float(trade.get("target_pct", 0.0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--token",   type=int, required=True, help="instrument token to backtest")
    parser.add_argument("--from",    dest="date_from", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to",      dest="date_to",   required=True, help="YYYY-MM-DD")
    parser.add_argument("--interval", default="5minute", choices=["3minute","5minute","10minute","15minute"])
    parser.add_argument("--side",    default="sell", choices=["buy","sell"])
    parser.add_argument("--qty",     type=int, default=1)
    args = parser.parse_args()

    api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)

    d0 = datetime.fromisoformat(args.date_from).date()
    d1 = datetime.fromisoformat(args.date_to).date()

    os.makedirs("reports/backtest", exist_ok=True)

    trades: List[Dict] = []
    equity_rows = []
    eq = 0.0

    for d in daterange(d0, d1):
        df = fetch_day_candles(kite, args.token, d, args.interval)
        row = simulate_day(
            df, d,
            side=args.side,
            qty=args.qty,
            entry_time=entry_time,
            end_time=end_time,
            sl_pct=sl_pct,
            target_pct=target_pct,
        )
        if row is None:
            continue
        trades.append(row)
        eq += row["pnl"]
        equity_rows.append({"date": d.isoformat(), "equity": round(eq, 2)})

    # Save outputs
    pd.DataFrame(trades).to_csv("reports/backtest/trades.csv", index=False)
    pd.DataFrame(equity_rows).to_csv("reports/backtest/equity_curve.csv", index=False)

    summary = {
        "trades": len(trades),
        "net_pnl": round(eq, 2),
        "win_rate": round(100.0 * (sum(t["pnl"] > 0 for t in trades) / max(1,len(trades))), 2),
        "config": {
            "entry_time": entry_time,
            "end_time": end_time,
            "sl_pct": sl_pct,
            "target_pct": target_pct,
            "interval": args.interval,
            "side": args.side,
            "qty": args.qty,
            "token": args.token,
            "range": [args.date_from, args.date_to],
        },
    }
    with open("reports/backtest/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("✅ Backtest done:",
          f"{summary['trades']} trades | Net PnL {summary['net_pnl']} | Win {summary['win_rate']}%")

if __name__ == "__main__":
    main()
