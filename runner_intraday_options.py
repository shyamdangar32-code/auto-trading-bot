#!/usr/bin/env python3
# runner_intraday_options.py
from __future__ import annotations
import os, json, argparse
from datetime import datetime, time
import pandas as pd
import numpy as np

# TA
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

# Our helpers
from bot.config import load_config
from bot.data_io import zerodha_prices
from bot.metrics import compute_metrics
from bot.evaluation import plot_equity_and_drawdown

LONG, SHORT, FLAT = 1, -1, 0

def within_trading_window(ts: pd.Timestamp, start_str: str, end_str: str) -> bool:
    t = ts.tz_convert("Asia/Kolkata").time()
    h1, m1 = map(int, start_str.split(":"))
    h2, m2 = map(int, end_str.split(":"))
    return (time(h1, m1) <= t <= time(h2, m2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load config ----
    cfg = load_config()
    icfg = cfg.get("intraday_options", {})
    capital = float(cfg.get("capital_rs", 100000))
    qty = int(icfg.get("order_qty", 1))

    # trading window & params
    start_t = icfg.get("start_time", "09:20")
    end_t   = icfg.get("end_time",   "15:15")
    ema_len = int(icfg.get("ma_len", 20))
    adx_len = int(icfg.get("adx_len", 14))
    atr_len = int(icfg.get("atr_len", 14))
    adx_min = int(icfg.get("adx_min", 10)) if "adx_min" in icfg else 10

    sl_mult   = float(icfg.get("sl_atr_mult", 1.0))
    tgt_rr    = float(icfg.get("tgt_rr", 1.5))
    trail_on  = bool(icfg.get("trail_start_atr", 1.0) is not None)
    trail_trg = float(icfg.get("trail_start_atr", 1.0))
    trail_mul = float(icfg.get("trail_atr_mult", 1.0))

    re_max    = int(icfg.get("reentry_max", 2))
    cooldown  = int(icfg.get("reentry_cooldown", 3))

    # ---- Fetch last 5d, 5m candles from Zerodha (index token) ----
    token = int(icfg.get("index_token", 260105))  # BANKNIFTY default
    df = zerodha_prices(token, period="5d", interval="5m")
    df = df.set_index(pd.DatetimeIndex(df["Date"])).sort_index()

    # ---- Indicators ----
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    ema   = EMAIndicator(close, window=ema_len).ema_indicator()
    adx   = ADXIndicator(high, low, close, window=adx_len).adx()
    atr   = AverageTrueRange(high, low, close, window=atr_len).average_true_range()

    d = pd.DataFrame({
        "Close": close,
        "High": high,
        "Low": low,
        "ema": ema,
        "adx": adx,
        "atr": atr,
    }).dropna()

    # basic cross conditions inside trading window
    d["cross_up"]   = (d["Close"].shift(1) <= d["ema"].shift(1)) & (d["Close"] > d["ema"])
    d["cross_down"] = (d["Close"].shift(1) >= d["ema"].shift(1)) & (d["Close"] < d["ema"])
    d["in_time"]    = [within_trading_window(ts, start_t, end_t) for ts in d.index]
    d["adx_ok"]     = d["adx"] >= adx_min

    # ---- Walk forward sim (index-level) ----
    position = FLAT
    entry_px = stop = target = np.nan
    last_exit_idx = -10**9
    re_count_today = 0
    eq_val = capital
    eq_curve = []
    trades = []

    idx = list(d.index)
    for i, ts in enumerate(idx):
        row = d.loc[ts]
        px  = float(row.Close)
        atr_val = float(row.atr)

        # day change => reset re-entries
        if i > 0 and ts.date() != idx[i-1].date():
            re_count_today = 0

        # trailing stop
        if position != FLAT and trail_on:
            if position == LONG:
                # trail only after profit >= trail_trg * ATR
                if (px - entry_px) >= trail_trg * atr_val:
                    new_stop = px - trail_mul * atr_val
                    stop = max(stop, new_stop)
            else:
                if (entry_px - px) >= trail_trg * atr_val:
                    new_stop = px + trail_mul * atr_val
                    stop = min(stop, new_stop)

        # exits
        did_exit = False
        if position == LONG:
            if row.Low <= stop:
                trades[-1]["exit_time"] = ts; trades[-1]["exit"] = stop
                trades[-1]["reason"] = "STOP"
                pnl = (stop - entry_px) * qty; trades[-1]["pnl"] = pnl
                eq_val += pnl; position = FLAT; did_exit = True
            elif row.High >= target:
                trades[-1]["exit_time"] = ts; trades[-1]["exit"] = target
                trades[-1]["reason"] = "TARGET"
                pnl = (target - entry_px) * qty; trades[-1]["pnl"] = pnl
                eq_val += pnl; position = FLAT; did_exit = True

        elif position == SHORT:
            if row.High >= stop:
                trades[-1]["exit_time"] = ts; trades[-1]["exit"] = stop
                trades[-1]["reason"] = "STOP"
                pnl = (entry_px - stop) * qty; trades[-1]["pnl"] = pnl
                eq_val += pnl; position = FLAT; did_exit = True
            elif row.Low <= target:
                trades[-1]["exit_time"] = ts; trades[-1]["exit"] = target
                trades[-1]["reason"] = "TARGET"
                pnl = (entry_px - target) * qty; trades[-1]["pnl"] = pnl
                eq_val += pnl; position = FLAT; did_exit = True

        if did_exit:
            last_exit_idx = i

        # entries
        if position == FLAT and d.loc[ts, "in_time"] and d.loc[ts, "adx_ok"]:
            ok_cd = (i - last_exit_idx) >= cooldown
            ok_re = (re_count_today < re_max) or (re_max == 0)
            if ok_cd and ok_re:
                if d.loc[ts, "cross_up"]:
                    position = LONG
                    entry_px = px
                    stop   = entry_px - sl_mult * atr_val
                    target = entry_px + tgt_rr * sl_mult * atr_val  # RR on ATR distance
                    trades.append({
                        "side": "LONG", "entry_time": ts, "entry": entry_px,
                        "exit_time": None, "exit": None, "reason": "", "pnl": 0.0
                    })
                    if last_exit_idx > -10**8: re_count_today += 1

                elif d.loc[ts, "cross_down"]:
                    position = SHORT
                    entry_px = px
                    stop   = entry_px + sl_mult * atr_val
                    target = entry_px - tgt_rr * sl_mult * atr_val
                    trades.append({
                        "side": "SHORT", "entry_time": ts, "entry": entry_px,
                        "exit_time": None, "exit": None, "reason": "", "pnl": 0.0
                    })
                    if last_exit_idx > -10**8: re_count_today += 1

        eq_curve.append(eq_val)

    # close any open at last bar price
    if position != FLAT:
        last_ts = idx[-1]
        last_px = float(d.loc[last_ts, "Close"])
        if position == LONG:
            pnl = (last_px - entry_px) * qty
        else:
            pnl = (entry_px - last_px) * qty
        trades[-1]["exit_time"] = last_ts
        trades[-1]["exit"] = last_px
        trades[-1]["reason"] = "EOD"
        trades[-1]["pnl"] = pnl
        eq_val += pnl
        eq_curve[-1] = eq_val  # update last point

    # ---- Persist artifacts (always) ----
    trades_df = pd.DataFrame(trades)
    equity_ser = pd.Series(eq_curve, index=d.index, name="equity")

    # metrics (even if zero trades/equity)
    summary = compute_metrics(trades_df if not trades_df.empty else pd.DataFrame(),
                              equity_ser if not equity_ser.empty else pd.Series([capital]),
                              starting_capital=capital)

    # write files
    trades_out = os.path.join(args.out_dir, "trades.csv")
    equity_out = os.path.join(args.out_dir, "equity.csv")
    metrics_out= os.path.join(args.out_dir, "metrics.json")
    trades_df.to_csv(trades_out, index=False)
    equity_ser.to_csv(equity_out, header=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # charts
    if not equity_ser.empty:
        plot_equity_and_drawdown(equity_ser, args.out_dir)

    print("✅ Intraday paper run done.")
    print("   Trades:", len(trades_df), "| ROI:", summary.get("roi_pct"), "%")
    print("   Files written to:", args.out_dir)

if __name__ == "__main__":
    main()
