# bot/strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import add_indicators  # adds: ema_fast, ema_slow, rsi, adx, atr

# ---- position constants ----
LONG: int = 1
SHORT: int = -1
FLAT: int = 0


def _entry_conditions(d: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series]:
    """
    Boolean series for long/short entries based on EMA + RSI + ADX.
    Expects columns: ema_fast, ema_slow, rsi, adx
    """
    adx_min = float(cfg.get("adx_min", 10))
    rsi_buy = float(cfg.get("rsi_buy", 30))
    rsi_sell = float(cfg.get("rsi_sell", 70))

    long_ok = (d["ema_fast"] > d["ema_slow"]) & (d["rsi"] >= rsi_buy) & (d["adx"] >= adx_min)
    short_ok = (d["ema_fast"] < d["ema_slow"]) & (d["rsi"] <= rsi_sell) & (d["adx"] >= adx_min)

    return long_ok.fillna(False), short_ok.fillna(False)


def prepare_signals(prices: pd.DataFrame, cfg: dict, **kwargs) -> pd.DataFrame:
    """
    Build indicators and generate entry signals.

    Extra kwargs are accepted to stay compatible with runners that pass
    optional flags (e.g., use_block=True). Unknown flags are ignored.

    Optional behaviour:
      - use_block (bool, default False): if True, suppress *same-direction*
        re-entries that arrive within 'block_bars' bars.
      - block_bars (int in cfg, default 0): minimum gap (bars) between two
        same-direction entries.
    """
    # Optional flags (ignored unless explicitly enabled)
    use_block: bool = bool(kwargs.get("use_block", False))
    block_bars: int = int(cfg.get("block_bars", 0))

    # 1) Indicators
    d = add_indicators(prices.copy(), cfg)

    # 2) Raw entry conditions
    long_ok, short_ok = _entry_conditions(d, cfg)

    # 3) Encode as signals
    d["long_entry"] = long_ok
    d["short_entry"] = short_ok
    d["signal"] = 0
    d.loc[d["long_entry"], "signal"] = LONG
    d.loc[d["short_entry"], "signal"] = SHORT

    # 4) Optional simple "blocking" to avoid too-dense signals
    if use_block and block_bars > 0:
        last_i = -10**9
        last_side = 0
        sig = d["signal"].to_numpy().copy()
        for i, s in enumerate(sig):
            if s != 0:
                # If same-side signal appears within block window, drop it
                if (i - last_i) < block_bars and s == last_side:
                    sig[i] = 0
                else:
                    last_i = i
                    last_side = s
        d["signal"] = sig
        d["long_entry"] = d["signal"].eq(LONG)
        d["short_entry"] = d["signal"].eq(SHORT)

    return d


def initial_stop_target(side: int, entry_price: float, atr: float, cfg: dict) -> tuple[float, float]:
    """
    Compute initial stop & target using ATR multiples.
    """
    stop_mult = float(cfg.get("stop_atr_mult", cfg.get("sl_atr_mult", 2.0)))
    take_mult = float(cfg.get("take_atr_mult", 3.0))

    if side == LONG:
        stop = entry_price - stop_mult * atr
        target = entry_price + take_mult * atr
    else:
        stop = entry_price + stop_mult * atr
        target = entry_price - take_mult * atr
    return float(stop), float(target)


def trail_stop(
    side: int,
    px: float,
    atr: float,
    curr_stop: float,
    entry: float,
    cfg: dict,
) -> float:
    """
    ATR trailing. Starts after unrealized > trail_start_atr * ATR.
    Supports cfg:
      - trailing_enabled (bool, default True)
      - trail_type ("atr", default "atr")
      - trail_start_atr (float, default 1.0)
      - trail_atr_mult (float, default 1.5)
    """
    if not cfg.get("trailing_enabled", True):
        return curr_stop
    if str(cfg.get("trail_type", "atr")).lower() != "atr":
        return curr_stop

    start_trigger = float(cfg.get("trail_start_atr", 1.0)) * atr
    trail_dist = float(cfg.get("trail_atr_mult", 1.5)) * atr

    if side == LONG:
        if (px - entry) >= start_trigger:
            new_stop = px - trail_dist
            return max(curr_stop, new_stop)
        return curr_stop
    elif side == SHORT:
        if (entry - px) >= start_trigger:
            new_stop = px + trail_dist
            return min(curr_stop, new_stop)
        return curr_stop
    else:
        return curr_stop
