# bot/strategy.py
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import time as dtime  # <-- for session window comparisons

# positions
LONG, SHORT, FLAT = 1, -1, 0

# ----------------- small utils -----------------
def _get(d: dict, key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default

def _cfg_block(cfg: dict, use_block: str) -> dict:
    """merge cfg[use_block] over top-level for convenience"""
    b = dict(cfg.get(use_block) or {})
    for k, v in cfg.items():
        if k not in b:
            b[k] = v
    # bubble nested dicts if present
    for k in ("exits", "reentry", "guardrails", "filters"):
        if k not in b and isinstance(cfg.get(use_block, {}), dict):
            b[k] = cfg.get(use_block, {}).get(k, {})
    return b

# ----------------- indicators ------------------
def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean().replace(0, np.nan)
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean().bfill()

def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Wilder's ADX"""
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0.0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0.0)) * down_move

    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    adx = dx.ewm(alpha=1/length, adjust=False).mean().fillna(0.0)
    return adx

def _within_session(ts: pd.Timestamp, start: str, end: str) -> bool:
    """
    Compare only on clock-time using datetime.time (no year/month/day needed).
    """
    t = (ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)).time()
    s_h, s_m = map(int, start.split(":"))
    e_h, e_m = map(int, end.split(":"))
    return dtime(s_h, s_m) <= t <= dtime(e_h, e_m)

# ----------------- PUBLIC API ------------------
def prepare_signals(prices: pd.DataFrame, cfg: dict, use_block: str = "backtest") -> pd.DataFrame:
    """
    Higher-quality entries for index-level intraday backtests.

    Filters:
      • Trend: EMA_fast vs EMA_slow
      • Strength: ADX >= adx_min
      • HTF confirmation: Close vs EMA(htf) on a higher timeframe (e.g., 15min)
      • Volatility: ATR% >= min_atr_pct
      • Session guard: only trade between session_start/end
      • Trigger: RSI cross around 50 (directional)
    """
    b = _cfg_block(cfg, use_block)
    f = dict(b.get("filters") or {})

    # params
    ema_fast_len = int(_get(b, "ema_fast", 21))
    ema_slow_len = int(_get(b, "ema_slow", 50))
    rsi_len      = int(_get(b, "rsi_len", 14))
    rsi_buy      = float(_get(b, "rsi_buy", 52.0))   # slightly stricter to boost win-rate
    rsi_sell     = float(_get(b, "rsi_sell", 48.0))
    atr_len      = int(_get(b, "atr_len", 14))
    poke_pct     = float(_get(b, "ema_poke_pct", 0.0001))  # tiny confirmation

    # filter params
    adx_len      = int(_get(f, "adx_len", 14))
    adx_min      = float(_get(f, "adx_min", 15))
    use_htf      = bool(_get(f, "use_htf", True))
    htf_rule     = str(_get(f, "htf_rule", "15min"))
    htf_ema_len  = int(_get(f, "htf_ema_len", 20))
    min_atr_pct  = float(_get(f, "min_atr_pct", 0.0003))  # 0.03%
    sess_start   = str(_get(f, "session_start", _get(b, "session_start", "09:20")))
    sess_end     = str(_get(f, "session_end",   _get(b, "session_end",   "15:20")))

    d = prices.copy()

    # base indicators
    d["ema_fast"] = _ema(d["Close"], ema_fast_len)
    d["ema_slow"] = _ema(d["Close"], ema_slow_len)
    d["rsi"]      = _rsi(d["Close"], rsi_len)
    d["atr"]      = _atr(d, atr_len)
    d["adx"]      = _adx(d, adx_len)

    # volatility %
    d["atr_pct"]  = (d["atr"] / d["Close"]).abs().fillna(0.0)

    # HTF EMA (confirm trend on higher timeframe)
    if use_htf:
        htf_close = d["Close"].resample(htf_rule).last().ffill()
        htf_ema   = _ema(htf_close, htf_ema_len)
        d["ema_htf"] = htf_ema.reindex(d.index, method="ffill")
    else:
        d["ema_htf"] = d["ema_slow"]

    # regime filters
    d["trend_long"]  = (d["ema_fast"] > d["ema_slow"])
    d["trend_short"] = (d["ema_fast"] < d["ema_slow"])
    d["htf_long"]    = (d["Close"] > d["ema_htf"])
    d["htf_short"]   = (d["Close"] < d["ema_htf"])
    d["strength_ok"] = (d["adx"] >= adx_min)
    d["vol_ok"]      = (d["atr_pct"] >= min_atr_pct)

    # tiny poke vs slow EMA
    d["poke_up"]   = d["Close"] > d["ema_slow"] * (1.0 + poke_pct)
    d["poke_down"] = d["Close"] < d["ema_slow"] * (1.0 - poke_pct)

    # RSI midline crosses (directional)
    d["rsi_cross_up"]   = (d["rsi"] > rsi_buy) & (d["rsi"].shift(1) <= rsi_buy)
    d["rsi_cross_down"] = (d["rsi"] < rsi_sell) & (d["rsi"].shift(1) >= rsi_sell)

    # session mask
    sess_mask = d.index.to_series().apply(lambda ts: _within_session(ts, sess_start, sess_end))

    # final entries = (trend + strength + HTF + volatility + poke) * trigger * session
    d["long_entry"] = (
        d["trend_long"] & d["strength_ok"] & d["htf_long"] & d["vol_ok"]
        & d["poke_up"] & d["rsi_cross_up"] & sess_mask
    ).fillna(False)

    d["short_entry"] = (
        d["trend_short"] & d["strength_ok"] & d["htf_short"] & d["vol_ok"]
        & d["poke_down"] & d["rsi_cross_down"] & sess_mask
    ).fillna(False)

    # clean head NaNs
    d = d.dropna(subset=["ema_fast","ema_slow","ema_htf","rsi","atr","adx"])
    return d

def initial_stop_target(side: int, entry_px: float, atr_val: float, exits_cfg: dict) -> tuple[float, float]:
    stop_mult = float((exits_cfg or {}).get("stop_atr_mult", 1.0))
    take_mult = float((exits_cfg or {}).get("take_atr_mult", 1.3))  # a tad higher R:R
    atr_val   = float(atr_val or 0.0)
    atr_val   = max(atr_val, 1e-6)

    if side == LONG:
        stop   = entry_px - stop_mult * atr_val
        target = entry_px + take_mult * atr_val
    else:
        stop   = entry_px + stop_mult * atr_val
        target = entry_px - take_mult * atr_val
    return float(stop), float(target)

def trail_stop(side: int, px: float, atr_val: float, stop: float, entry_px: float, exits_cfg: dict) -> float:
    trail_block = (exits_cfg or {}).get("trail", {}) if isinstance(exits_cfg, dict) else {}
    t_mult = float(trail_block.get("atr_mult", 1.0))
    atr_val = float(atr_val or 0.0)
    atr_val = max(atr_val, 1e-6)
    if side == LONG:
        return float(max(stop, px - t_mult * atr_val))
    else:
        return float(min(stop, px + t_mult * atr_val))
