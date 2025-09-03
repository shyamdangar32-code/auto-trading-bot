# bot/strategy.py 
from __future__ import annotations
import numpy as np
import pandas as pd

# simple constants used by backtest.py
LONG, SHORT, FLAT = 1, -1, 0

# --- helpers ---------------------------------------------------------------

def _get(block: dict, key: str, default):
    try:
        return block.get(key, default)
    except Exception:
        return default

def _cfg_block(cfg: dict, use_block: str) -> dict:
    """
    Backtest calls prepare_signals(prices, cfg, use_block="backtest").
    We merge:
      1) cfg[use_block] (if present)
      2) top-level cfg (fallback)
    so either place works.
    """
    b = dict(cfg.get(use_block) or {})
    for k, v in cfg.items():
        if k not in b:
            b[k] = v
    # nested sub-blocks (backtest.exits / backtest.reentry / backtest.guardrails)
    for k in ("exits", "reentry", "guardrails"):
        if k not in b and isinstance(cfg.get(use_block, {}), dict):
            b[k] = cfg.get(use_block, {}).get(k, {})
    return b

# --- indicators ------------------------------------------------------------

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # prevent div by zero
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean().fillna(method="bfill")

# --- PUBLIC API used by backtest.py ---------------------------------------

def prepare_signals(prices: pd.DataFrame, cfg: dict, use_block: str = "backtest") -> pd.DataFrame:
    """
    Generate frequent but reasonable intraday signals on index candles.

    Logic (deliberately *loosened* to increase trade count):
      • Trend filter: EMA_fast vs EMA_slow (state).
      • Entry trigger: RSI crosses midline (50) in the direction of trend.
      • Optional poke filter: price above/below EMA_slow by tiny % (ema_poke_pct).

    Returned columns expected by backtest:
      'long_entry', 'short_entry', 'atr' (and standard OHLC).
    """
    b = _cfg_block(cfg, use_block)

    ema_fast_len = int(_get(b, "ema_fast", 21))
    ema_slow_len = int(_get(b, "ema_slow", 50))
    rsi_len      = int(_get(b, "rsi_len", 14))
    rsi_buy      = float(_get(b, "rsi_buy", 50.0))
    rsi_sell     = float(_get(b, "rsi_sell", 50.0))
    atr_len      = int(_get(b, "atr_len", 14))
    poke_pct     = float(_get(b, "ema_poke_pct", 0.0001))  # 0.01%

    d = prices.copy()
    # basic indicators
    d["ema_fast"] = _ema(d["Close"], ema_fast_len)
    d["ema_slow"] = _ema(d["Close"], ema_slow_len)
    d["rsi"] = _rsi(d["Close"], rsi_len)
    d["atr"] = _atr(d, atr_len)

    # trend regime
    d["trend_long"]  = d["ema_fast"] > d["ema_slow"]
    d["trend_short"] = d["ema_fast"] < d["ema_slow"]

    # small poke relative to EMA_slow to avoid whips
    d["poke_up"]   = d["Close"] > d["ema_slow"] * (1.0 + poke_pct)
    d["poke_down"] = d["Close"] < d["ema_slow"] * (1.0 - poke_pct)

    # RSI midline crosses
    d["rsi_cross_up"]   = (d["rsi"] > rsi_buy) & (d["rsi"].shift(1) <= rsi_buy)
    d["rsi_cross_down"] = (d["rsi"] < rsi_sell) & (d["rsi"].shift(1) >= rsi_sell)

    # final entries (loosened): trend + (RSI cross) + light poke
    d["long_entry"]  = (d["trend_long"]  & d["rsi_cross_up"]   & d["poke_up"]).fillna(False)
    d["short_entry"] = (d["trend_short"] & d["rsi_cross_down"] & d["poke_down"]).fillna(False)

    # Clean NaNs at start
    d = d.dropna(subset=["ema_fast", "ema_slow", "rsi", "atr"])
    return d

def initial_stop_target(side: int, entry_px: float, atr_val: float, exits_cfg: dict) -> tuple[float, float]:
    stop_mult   = float((exits_cfg or {}).get("stop_atr_mult", 1.0))
    take_mult   = float((exits_cfg or {}).get("take_atr_mult", 1.2))
    atr_val     = float(atr_val or 0.0)
    atr_val     = max(atr_val, 1e-6)  # guard

    if side == LONG:
        stop   = entry_px - stop_mult * atr_val
        target = entry_px + take_mult * atr_val
    else:  # SHORT
        stop   = entry_px + stop_mult * atr_val
        target = entry_px - take_mult * atr_val
    return float(stop), float(target)

def trail_stop(side: int, px: float, atr_val: float, stop: float, entry_px: float, exits_cfg: dict) -> float:
    """
    Simple ATR trailing stop.
    LONG:  max(old_stop, px - trail_mult * ATR)
    SHORT: min(old_stop, px + trail_mult * ATR)
    """
    trail_block = (exits_cfg or {}).get("trail", {}) if isinstance(exits_cfg, dict) else {}
    t_mult = float(trail_block.get("atr_mult", 1.0))
    atr_val = float(atr_val or 0.0)
    atr_val = max(atr_val, 1e-6)

    if side == LONG:
        new_stop = px - t_mult * atr_val
        return float(max(stop, new_stop))
    else:
        new_stop = px + t_mult * atr_val
        return float(min(stop, new_stop))
