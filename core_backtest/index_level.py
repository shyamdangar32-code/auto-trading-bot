# core_backtest/index_level.py
from __future__ import annotations
import numpy as np
import pandas as pd

from bot.strategy import prepare_signals, initial_stop_target, trail_stop, LONG, SHORT, FLAT
from bot.metrics import compute_metrics

def run(prices: pd.DataFrame, cfg: dict):
    """
    Fast index/futures-style backtest on underlying OHLC.
    Uses EMA/RSI/ADX entries with ATR-based SL, R-multiple targets,
    optional ATR trailing, re-entries with cooldown.
    Returns: (summary, trades_df, equity_series)
    """
    d = prepare_signals(prices, {
        "ema_fast": cfg["entry"]["ema_fast"],
        "ema_slow": cfg["entry"]["ema_slow"],
        "rsi_len":  cfg["entry"]["rsi_len"],
        "rsi_buy":  cfg["entry"]["rsi_buy"],
        "rsi_sell": cfg["entry"]["rsi_sell"],
        "adx_len":  cfg["entry"]["adx_len"],
        "atr_len":  cfg["entry"]["atr_len"],
        "adx_min":  cfg["entry"]["adx_min"],
        "stop_atr_mult": cfg["risk"]["stop_atr_mult"],
        "take_atr_mult": cfg["risk"]["take_rr"] * cfg["risk"]["stop_atr_mult"],
        "trailing_enabled": cfg["trailing"]["enabled"],
        "trail_type": cfg["trailing"]["type"],
        "trail_start_atr": cfg["trailing"]["trail_start_atr"],
        "trail_atr_mult": cfg["trailing"]["trail_atr_mult"],
    })

    qty        = int(cfg.get("order_qty", 1))
    capital    = float(cfg.get("capital_rs", 100000.0))
    re_max     = int(cfg["reentry"]["max_per_day"])
    cooldown   = int(cfg["reentry"]["cooldown_bars"])

    position = FLAT
    entry_px = stop = target = np.nan
    re_count = 0
    last_exit_idx = -10**9
    trades = []

    eq = capital
    eq_curve = []

    idx = list(d.index)
    for i, ts in enumerate(idx):
        row = d.loc[ts]
        px  = float(row["Close"])
        atr = float(row.get("atr", np.nan)) if not np.isnan(row.get("atr", np.nan)) else 0.0

        # trail
        if position != FLAT:
            stop = trail_stop(position, px, atr, stop, entry_px, {
                "trailing_enabled": cfg["trailing"]["enabled"],
                "trail_type": cfg["trailing"]["type"],
                "trail_start_atr": cfg["trailing"]["trail_start_atr"],
                "trail_atr_mult": cfg["trailing"]["trail_atr_mult"],
            })

        # exits
        did_exit = False
        if position == LONG:
            if row["Low"] <= stop:
                pnl = (stop - entry_px) * qty
                trades[-1].update(exit_time=ts, exit=stop, reason="STOP", pnl=pnl)
                position = FLAT; eq += pnl; did_exit = True
            elif row["High"] >= target:
                pnl = (target - entry_px) * qty
                trades[-1].update(exit_time=ts, exit=target, reason="TARGET", pnl=pnl)
                position = FLAT; eq += pnl; did_exit = True

        elif position == SHORT:
            if row["High"] >= stop:
                pnl = (entry_px - stop) * qty
                trades[-1].update(exit_time=ts, exit=stop, reason="STOP", pnl=pnl)
                position = FLAT; eq += pnl; did_exit = True
            elif row["Low"] <= target:
                pnl = (entry_px - target) * qty
                trades[-1].update(exit_time=ts, exit=target, reason="TARGET", pnl=pnl)
                position = FLAT; eq += pnl; did_exit = True

        if did_exit:
            last_exit_idx = i

        # entries & re-entries
        if position == FLAT:
            ok = (i - last_exit_idx) >= cooldown
            if ok:
                if bool(row.get("long_entry", False)) and (re_count < re_max or re_max == 0):
                    position = LONG
                    entry_px = px
                    stop, target = initial_stop_target(LONG, entry_px, atr, {
                        "stop_atr_mult": cfg["risk"]["stop_atr_mult"],
                        "take_atr_mult": cfg["risk"]["take_rr"] * cfg["risk"]["stop_atr_mult"],
                    })
                    trades.append(dict(side="LONG", entry_time=ts, entry=entry_px))
                    re_count += 1 if last_exit_idx > -10**8 else 0

                elif bool(row.get("short_entry", False)) and (re_count < re_max or re_max == 0):
                    position = SHORT
                    entry_px = px
                    stop, target = initial_stop_target(SHORT, entry_px, atr, {
                        "stop_atr_mult": cfg["risk"]["stop_atr_mult"],
                        "take_atr_mult": cfg["risk"]["take_rr"] * cfg["risk"]["stop_atr_mult"],
                    })
                    trades.append(dict(side="SHORT", entry_time=ts, entry=entry_px))
                    re_count += 1 if last_exit_idx > -10**8 else 0

        # reset day
        if i > 0:
            if pd.Timestamp(idx[i-1]).date() != pd.Timestamp(ts).date():
                re_count = 0

        eq_curve.append(eq)

    trades_df = pd.DataFrame(trades)
    equity = pd.Series(eq_curve, index=d.index, name="equity")
    summary = compute_metrics(trades_df, equity, capital)
    return summary, trades_df, equity
