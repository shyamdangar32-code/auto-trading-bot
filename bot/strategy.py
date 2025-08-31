# bot/strategy.py
from __future__ import annotations
import numpy as np
import pandas as pd
from .indicators import add_indicators  # uses ta to add ema/rsi/adx/atr

LONG = 1
SHORT = -1
FLAT = 0

def _entry_conditions(d: pd.DataFrame, cfg: dict):
    """
    Boolean series for long/short entries.
    We support three modes to control trade frequency:
      - strict:   old logic (EMA trend + RSI + ADX)
      - balanced: EMA trend AND (RSI momentum OR price>EMA_fast), lower ADX
      - aggressive: (EMA cross OR RSI momentum), no ADX requirement
    """
    mode = (cfg.get("signal_mode") or "balanced").lower()
    adx_min_strict = cfg.get("adx_min_strict", 18)
    adx_min_bal    = cfg.get("adx_min_bal", 10)

    ema_f = d["ema_fast"]
    ema_s = d["ema_slow"]
    rsi   = d["rsi"]
    adx   = d["adx"]
    px    = d["Close"]

    # momentum thresholds (slightly loose to get more trades)
    rsi_buy  = float(cfg.get("rsi_buy", 52))   # long if RSI > 52
    rsi_sell = float(cfg.get("rsi_sell", 48))  # short if RSI < 48

    # price poke above/below fast EMA (tiny buffer ~0.05%)
    poke = cfg.get("ema_poke_pct", 0.0005)
    above_f = px > (ema_f * (1.0 + poke))
    below_f = px < (ema_f * (1.0 - poke))

    if mode == "strict":
        long_ok  = (ema_f > ema_s) & (rsi >= rsi_buy)  & (adx >= adx_min_strict)
        short_ok = (ema_f < ema_s) & (rsi <= rsi_sell) & (adx >= adx_min_strict)

    elif mode == "aggressive":
        # Either EMA cross OR RSI momentum, ignore ADX to get more signals
        long_ok  = ((ema_f > ema_s) | (above_f)) & (rsi >= rsi_buy)
        short_ok = ((ema_f < ema_s) | (below_f)) & (rsi <= rsi_sell)

    else:  # "balanced" (default)
        # EMA trend and (momentum from RSI OR small poke above/below fast EMA)
        long_ok  = (ema_f > ema_s) & ((rsi >= rsi_buy) | (above_f)) & (adx >= adx_min_bal)
        short_ok = (ema_f < ema_s) & ((rsi <= rsi_sell) | (below_f)) & (adx >= adx_min_bal)

    return long_ok.fillna(False), short_ok.fillna(False)


def prepare_signals(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Returns a dataframe with indicators and 'long_entry', 'short_entry'.
    Expected columns: ['Open','High','Low','Close']
    """
    d = add_indicators(prices.copy(), cfg)
    long_ok, short_ok = _entry_conditions(d, cfg)
    d["long_entry"] = long_ok
    d["short_entry"] = short_ok
    d["signal"] = 0
    d.loc[d["long_entry"], "signal"] = LONG
    d.loc[d["short_entry"], "signal"] = SHORT
    return d


def initial_stop_target(side: int, entry_price: float, atr: float, cfg: dict):
    stop_mult = float(cfg.get("stop_atr_mult", 1.2))   # tighter default
    take_mult = float(cfg.get("take_atr_mult", 1.6))   # modest take to book more trades
    if side == LONG:
        stop   = entry_price - stop_mult * atr
        target = entry_price + take_mult * atr
    else:
        stop   = entry_price + stop_mult * atr
        target = entry_price - take_mult * atr
    return float(stop), float(target)


def trail_stop(side: int, px: float, atr: float, curr_stop: float, entry: float, cfg: dict):
    """
    ATR trailing. Starts earlier to keep trades active but protected.
    """
    if not cfg.get("trailing_enabled", True):
        return curr_stop
    if cfg.get("trail_type", "atr") != "atr":
        return curr_stop

    start_trigger = float(cfg.get("trail_start_atr", 0.6)) * atr   # start earlier
    trail_dist    = float(cfg.get("trail_atr_mult", 1.0)) * atr    # closer trail

    if side == LONG:
        if (px - entry) >= start_trigger:
            new_stop = px - trail_dist
            return max(curr_stop, new_stop) if not np.isnan(curr_stop) else new_stop
        return curr_stop
    else:
        if (entry - px) >= start_trigger:
            new_stop = px + trail_dist
            return min(curr_stop, new_stop) if not np.isnan(curr_stop) else new_stop
        return curr_stop
