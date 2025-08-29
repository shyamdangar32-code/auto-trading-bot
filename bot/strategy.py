# bot/strategy.py
import numpy as np
import pandas as pd
from .indicators import add_indicators  # uses ta to add ema/rsi/adx/atr

LONG  = 1
SHORT = -1
FLAT  = 0

def _entry_conditions(d: pd.DataFrame, cfg: dict):
    """
    Boolean series for long/short entries based on EMA + RSI + ADX.
    """
    adx_min = float(cfg.get("adx_min", 10))
    rsi_buy = float(cfg.get("rsi_buy", 30))
    rsi_sell = float(cfg.get("rsi_sell", 70))

    long_ok  = (d["ema_fast"] > d["ema_slow"]) & (d["rsi"] >= rsi_buy)  & (d["adx"] >= adx_min)
    short_ok = (d["ema_fast"] < d["ema_slow"]) & (d["rsi"] <= rsi_sell) & (d["adx"] >= adx_min)
    return long_ok.fillna(False), short_ok.fillna(False)

def prepare_signals(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Returns a dataframe with indicators and entry flags.
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
    stop_mult = float(cfg.get("stop_atr_mult", 2.0))
    take_mult = float(cfg.get("take_atr_mult", 3.0))
    if side == LONG:
        stop   = entry_price - stop_mult * atr
        target = entry_price + take_mult * atr
    else:
        stop   = entry_price + stop_mult * atr
        target = entry_price - take_mult * atr
    return float(stop), float(target)

def trail_stop(side: int, px: float, atr: float, curr_stop: float, entry: float, cfg: dict):
    """
    ATR trailing. Starts after unrealized > trail_start_atr * ATR.
    """
    if not cfg.get("trailing_enabled", True):
        return curr_stop
    if cfg.get("trail_type", "atr") != "atr":
        return curr_stop

    start_trigger = float(cfg.get("trail_start_atr", 1.0)) * atr
    trail_dist    = float(cfg.get("trail_atr_mult", 1.5)) * atr

    if side == LONG:
        if (px - entry) >= start_trigger:
            new_stop = px - trail_dist
            return max(curr_stop, new_stop)
        return curr_stop
    else:
        if (entry - px) >= start_trigger:
            new_stop = px + trail_dist
            return min(curr_stop, new_stop)
        return curr_stop
