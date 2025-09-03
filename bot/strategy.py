# bot/strategy.py
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import time as dtime  # session compare

# position enums
LONG, SHORT, FLAT = 1, -1, 0


# --------- small helpers ----------
def _get(d: dict, k: str, default=None):
    try:
        return d.get(k, default)
    except Exception:
        return default

def _cfg_block(cfg: dict, use_block: str) -> dict:
    """
    Merge block over root so both root keys and block keys are available.
    Also lift common nested dicts if present.
    """
    b = dict(cfg.get(use_block) or {})
    for k, v in cfg.items():
        if k not in b:
            b[k] = v
    for k in ("exits", "reentry", "guardrails", "filters"):
        if k not in b and isinstance(cfg.get(use_block, {}), dict):
            b[k] = cfg.get(use_block, {}).get(k, {})
    return b


# --------- indicators (vectorized, light) ----------
def _ema(x: pd.Series, length: int) -> pd.Series:
    return x.ewm(span=length, adjust=False).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    up_ema = up.ewm(alpha=1/length, adjust=False).mean()
    dn_ema = dn.ewm(alpha=1/length, adjust=False).mean().replace(0, np.nan)
    rs = up_ema / dn_ema
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean().bfill()

def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm  = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean().replace(0, np.nan)

    plus_di  = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100.0
    return dx.ewm(alpha=1/length, adjust=False).mean().fillna(0.0)

def _within_session(ts: pd.Timestamp, start: str, end: str) -> bool:
    """Clock-time compare using datetime.time (no year/month/day required)."""
    t = (ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)).time()
    s_h, s_m = map(int, start.split(":"))
    e_h, e_m = map(int, end.split(":"))
    return dtime(s_h, s_m) <= t <= dtime(e_h, e_m)


# --------- public API ----------
def prepare_signals(prices: pd.DataFrame, cfg: dict, use_block: str = "backtest") -> pd.DataFrame:
    """
    Higher-quality intraday index signals.

    Entry (long):
      Trend:      EMA_fast > EMA_slow  AND Close > EMA_HTF
      Strength:   ADX >= adx_min
      Volatility: ATR% >= min_atr_pct  (avoid chop)
      Liquidity:  Volume >= vol_ema * vol_k
      Trigger:    (1) RSI crosses above rsi_buy  AND
                  (2) EMA_fast crosses above EMA_slow (golden cross)
                  (3) 'poke' above EMA_slow by tiny pct
      Session:    within session_start..end

    Short is symmetric.
    """
    b = _cfg_block(cfg, use_block)
    f = dict(b.get("filters") or {})

    # ---- core params ----
    ema_fast_len = int(_get(b, "ema_fast", 21))
    ema_slow_len = int(_get(b, "ema_slow", 50))
    rsi_len      = int(_get(b, "rsi_len", 14))
    rsi_buy      = float(_get(b, "rsi_buy", 52.0))     # stricter than 50
    rsi_sell     = float(_get(b, "rsi_sell", 48.0))
    atr_len      = int(_get(b, "atr_len", 14))
    poke_pct     = float(_get(b, "ema_poke_pct", 0.0002))  # 0.02%

    # ---- filter params ----
    adx_len      = int(_get(f, "adx_len", 14))
    adx_min      = float(_get(f, "adx_min", 15))
    use_htf      = bool(_get(f, "use_htf", True))
    htf_rule     = str(_get(f, "htf_rule", "15min"))
    htf_ema_len  = int(_get(f, "htf_ema_len", 20))
    min_atr_pct  = float(_get(f, "min_atr_pct", 0.00025))  # 0.025%
    sess_start   = str(_get(f, "session_start", _get(b, "session_start", "09:20")))
    sess_end     = str(_get(f, "session_end",   _get(b, "session_end",   "15:20")))
    vol_ema_len  = int(_get(f, "vol_ema_len", 20))
    vol_k        = float(_get(f, "vol_k", 0.9))  # allow ≥90% of recent average volume

    d = prices.copy()

    # ---- base indicators ----
    d["ema_fast"] = _ema(d["Close"], ema_fast_len)
    d["ema_slow"] = _ema(d["Close"], ema_slow_len)
    d["rsi"]      = _rsi(d["Close"], rsi_len)
    d["atr"]      = _atr(d, atr_len)
    d["adx"]      = _adx(d, adx_len)
    d["atr_pct"]  = (d["atr"] / d["Close"]).abs().fillna(0.0)

    # volume EMA (robust across symbols; if Volume missing, create zeros)
    if "Volume" not in d.columns:
        d["Volume"] = 0.0
    d["vol_ema"] = d["Volume"].ewm(span=vol_ema_len, adjust=False).mean().replace(0, np.nan)

    # ---- higher timeframe confirmation ----
    if use_htf:
        htf_close = d["Close"].resample(htf_rule).last().ffill()
        htf_ema   = _ema(htf_close, htf_ema_len)
        d["ema_htf"] = htf_ema.reindex(d.index, method="ffill")
    else:
        d["ema_htf"] = d["ema_slow"]

    # ---- regime masks ----
    d["trend_long"]   = (d["ema_fast"] > d["ema_slow"])
    d["trend_short"]  = (d["ema_fast"] < d["ema_slow"])
    d["htf_long"]     = (d["Close"] > d["ema_htf"])
    d["htf_short"]    = (d["Close"] < d["ema_htf"])
    d["strength_ok"]  = (d["adx"] >= adx_min)
    d["vol_ok"]       = (d["atr_pct"] >= min_atr_pct)
    d["liq_ok"]       = (d["Volume"] >= (d["vol_ema"] * vol_k).fillna(0))

    # tiny poke vs slow EMA (reduces early fakeouts)
    d["poke_up"]      = d["Close"] > d["ema_slow"] * (1.0 + poke_pct)
    d["poke_down"]    = d["Close"] < d["ema_slow"] * (1.0 - poke_pct)

    # EMA cross triggers
    d["gc"]  = (d["ema_fast"] > d["ema_slow"]) & (d["ema_fast"].shift(1) <= d["ema_slow"].shift(1))
    d["dc"]  = (d["ema_fast"] < d["ema_slow"]) & (d["ema_fast"].shift(1) >= d["ema_slow"].shift(1))

    # RSI midline directional crosses (with hysteresis)
    d["rsi_up"]   = (d["rsi"] > rsi_buy) & (d["rsi"].shift(1) <= rsi_buy)
    d["rsi_down"] = (d["rsi"] < rsi_sell) & (d["rsi"].shift(1) >= rsi_sell)

    # session mask
    sess_mask = d.index.to_series().apply(lambda ts: _within_session(ts, sess_start, sess_end))

    # ---- final entries ----
    long_base = (
        d["trend_long"] & d["htf_long"] & d["strength_ok"] & d["vol_ok"] & d["liq_ok"]
    )
    short_base = (
        d["trend_short"] & d["htf_short"] & d["strength_ok"] & d["vol_ok"] & d["liq_ok"]
    )

    d["long_entry"] = (
        long_base
        & d["poke_up"]
        & d["gc"]
        & d["rsi_up"]
        & sess_mask
    ).fillna(False)

    d["short_entry"] = (
        short_base
        & d["poke_down"]
        & d["dc"]
        & d["rsi_down"]
        & sess_mask
    ).fillna(False)

    # drop head NaNs to align backtester loops
    d = d.dropna(subset=["ema_fast", "ema_slow", "ema_htf", "rsi", "atr", "adx"])
    return d


def initial_stop_target(side: int, entry_px: float, atr_val: float, exits_cfg: dict) -> tuple[float, float]:
    """
    ATR-based stop/target. Slightly asymmetric default to lift R:R.
    """
    stop_mult = float((exits_cfg or {}).get("stop_atr_mult", 1.0))
    take_mult = float((exits_cfg or {}).get("take_atr_mult", 1.3))
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
    """
    Simple ATR trailing — keeps walking the stop in trade direction.
    """
    trail_block = (exits_cfg or {}).get("trail", {}) if isinstance(exits_cfg, dict) else {}
    t_mult = float(trail_block.get("atr_mult", 1.0))
    atr_val = float(atr_val or 0.0)
    atr_val = max(atr_val, 1e-6)

    if side == LONG:
        return float(max(stop, px - t_mult * atr_val))
    else:
        return float(min(stop, px + t_mult * atr_val))
