# bot/strategy.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

# ---- try using repo's indicators helpers; fall back to local impls ----
try:
    # your repo already has these
    from .indicators import ema, rsi, atr, adx  # type: ignore
except Exception:
    # lightweight fallbacks (if import path differs)
    def ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0).rolling(length).mean()
        down = -delta.clip(upper=0).rolling(length).mean()
        rs = up / (down.replace(0, np.nan))
        out = 100 - (100 / (1 + rs))
        return out.fillna(50)

    def atr(h: pd.Series, l: pd.Series, c: pd.Series, length: int = 14) -> pd.Series:
        hl = (h - l).abs()
        hc = (h - c.shift()).abs()
        lc = (l - c.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    def adx(h: pd.Series, l: pd.Series, c: pd.Series, length: int = 14) -> pd.Series:
        # very compact ADX approximation
        up = h.diff()
        down = -l.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = atr(h, l, c, 1)
        plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(span=length, adjust=False).mean() / tr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(span=length, adjust=False).mean() / tr.replace(0, np.nan)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
        return dx.ewm(span=length, adjust=False).mean().fillna(20.0)


Profile = Literal["loose", "medium", "strict"]


@dataclass
class RiskPlan:
    risk_perc: float     # capital risk per trade (e.g. 0.005 = 0.5%)
    sl_atr_mult: float
    tp_atr_mult: float
    ema_fast: int
    ema_slow: int
    rsi_len: int
    adx_min: float       # min ADX to allow momentum entries
    cooldown: int        # bars to cool-down after exit


PROFILES: Dict[Profile, RiskPlan] = {
    "loose":  RiskPlan(0.01, 1.2, 2.4, 9, 21, 10, 12.0, 3),
    "medium": RiskPlan(0.0075, 1.0, 2.0, 10, 30, 14, 14.0, 5),
    "strict": RiskPlan(0.005, 0.9, 1.8, 12, 40, 14, 16.0, 7),
}


def _position_size(entry_px: float, sl_px: float, capital: float, min_qty: int) -> int:
    """Risk-based position sizing; falls back to min_qty when computation invalid."""
    risk_per_share = abs(entry_px - sl_px)
    if not np.isfinite(risk_per_share) or risk_per_share <= 0:
        return max(1, min_qty)
    risk_budget = max(0.0, capital)
    # qty so that max loss ~= risk_perc * capital
    qty = int(math.floor((0.01 * risk_budget) / risk_per_share))  # 1% *capital per signal if caller didn’t override
    return max(min_qty, qty)


def prepare_signals(
    prices: pd.DataFrame,
    cfg: Dict,
    profile: Profile = "medium",
) -> pd.DataFrame:
    """
    Build rule-based signals + SL/TP/pos_size columns.
    Expected price columns: ['open','high','low','close'] (case-insensitive ok).
    """
    df = prices.copy()

    # normalize column names
    rename_map = {c: c.lower() for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # indicators
    plan = PROFILES.get(profile, PROFILES["medium"])
    df["ema_f"] = ema(df["close"], plan.ema_fast)
    df["ema_s"] = ema(df["close"], plan.ema_slow)
    df["rsi"] = rsi(df["close"], plan.rsi_len)
    df["atr"] = atr(df["high"], df["low"], df["close"], 14)
    df["adx"] = adx(df["high"], df["low"], df["close"], 14)

    # momentum/mean-reversion blended logic
    long_cond = (df["ema_f"] > df["ema_s"]) & (df["rsi"].between(45, 75)) & (df["adx"] >= plan.adx_min)
    short_cond = (df["ema_f"] < df["ema_s"]) & (df["rsi"].between(25, 55)) & (df["adx"] >= plan.adx_min)

    # signal (+1/-1/0) with basic de-whipsaw filter (no flip inside same bar range)
    df["signal"] = 0
    df.loc[long_cond, "signal"] = 1
    df.loc[short_cond, "signal"] = -1

    # build SL/TP from ATR
    sl_long = df["close"] - plan.sl_atr_mult * df["atr"]
    tp_long = df["close"] + plan.tp_atr_mult * df["atr"]
    sl_short = df["close"] + plan.sl_atr_mult * df["atr"]
    tp_short = df["close"] - plan.tp_atr_mult * df["atr"]

    df["sl_px"] = np.where(df["signal"] > 0, sl_long, np.where(df["signal"] < 0, sl_short, np.nan))
    df["tp_px"] = np.where(df["signal"] > 0, tp_long, np.where(df["signal"] < 0, tp_short, np.nan))

    # optional cool-down after an exit to avoid overtrading; we only stamp the field here
    df["cooldown_bars"] = plan.cooldown

    # risk-based pos_size (engine will use it if provided)
    capital = float(cfg.get("capital_rs", 100000.0))
    fallback_qty = int(cfg.get("order_qty", 1))
    # compute a size per row; realistic engines pick only when the trade triggers
    entry_price = df["close"]
    est_sl = df["sl_px"]
    df["pos_size"] = [
        _position_size(ep, sp, capital * plan.risk_perc, fallback_qty)
        if np.isfinite(ep) and np.isfinite(sp) else fallback_qty
        for ep, sp in zip(entry_price, est_sl)
    ]

    # clean NaNs at the start
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["ema_f", "ema_s", "rsi", "atr"], inplace=True)

    return df
