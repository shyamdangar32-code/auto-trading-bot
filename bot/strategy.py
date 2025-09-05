# bot/strategy.py
from __future__ import annotations

import numpy as np
import pandas as pd

from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# -------------------------
# Small TA helpers
# -------------------------

def _ema(series: pd.Series, n: int) -> pd.Series:
    return EMAIndicator(close=series, window=n).ema_indicator()

def _rsi(series: pd.Series, n: int) -> pd.Series:
    return RSIIndicator(close=series, window=n).rsi()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return ADXIndicator(high=high, low=low, close=close, window=n).adx()

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return AverageTrueRange(high=high, low=low, close=close, window=n).average_true_range()

def _slope(x: pd.Series, lookback: int = 5) -> pd.Series:
    # simple slope proxy
    return x.diff(lookback)


# -------------------------
# Higher timeframe context
# -------------------------

def _resample_htf(df: pd.DataFrame, rule: str, ema_len: int) -> pd.Series:
    """
    Resample into a higher timeframe and mark bars where HTF close > HTF EMA.
    Returned series is aligned to original df.index via ffill.
    """
    ohlc = (
        df[["Open", "High", "Low", "Close"]]
        .resample(rule, label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    htf_ema = _ema(ohlc["Close"], ema_len)
    htf_ok = (ohlc["Close"] > htf_ema).reindex(df.index, method="ffill").fillna(False)
    return htf_ok.astype(bool)


# -------------------------
# Profiles (loose/medium/strict)
# -------------------------

def profile_presets(profile: str) -> dict:
    p = (profile or "loose").lower()
    if p == "strict":
        return dict(
            adx_min=20, rsi_buy=55, rsi_sell=45,
            ema_slope_lb=5, min_atr_pct=0.0004,
            htf_rule="15min"
        )
    if p == "medium":
        return dict(
            adx_min=16, rsi_buy=53, rsi_sell=47,
            ema_slope_lb=4, min_atr_pct=0.00035,
            htf_rule="15min"
        )
    # loose (default)
    return dict(
        adx_min=12, rsi_buy=52, rsi_sell=48,
        ema_slope_lb=3, min_atr_pct=0.00030,
        htf_rule="15min"
    )


# -------------------------
# Signal preparation
# -------------------------

def prepare_signals(prices: pd.DataFrame, cfg: dict, profile: str = "loose") -> pd.DataFrame:
    """
    Builds indicator columns + boolean entry/exit hints expected by the backtest engine.
    Returns a new DataFrame aligned/sorted by index.
    """
    df = prices.copy().sort_index()
    presets = profile_presets(profile)

    # ---- parameters (can be overridden via cfg["backtest"]) ----
    b = (cfg.get("backtest") or {})
    filters = (b.get("filters") or {})

    ema_fast = int(b.get("ema_fast", cfg.get("ema_fast", 21)))
    ema_slow = int(b.get("ema_slow", cfg.get("ema_slow", 50)))
    rsi_len  = int(b.get("rsi_len", 14))
    adx_len  = int(filters.get("adx_len", 14))
    atr_len  = int(b.get("atr_len", cfg.get("atr_len", 14)))
    ema_poke = float(b.get("ema_poke_pct", cfg.get("ema_poke_pct", 1e-4)))

    # ---- indicators ----
    df["ema_fast"] = _ema(df["Close"], ema_fast)
    df["ema_slow"] = _ema(df["Close"], ema_slow)
    df["rsi"]      = _rsi(df["Close"], rsi_len)
    df["adx"]      = _adx(df["High"], df["Low"], df["Close"], adx_len)
    df["atr"]      = _atr(df["High"], df["Low"], df["Close"], atr_len)

    # clean ATR to avoid NaN/inf downstream
    df["atr"] = df["atr"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["atr_pct"] = (df["atr"] / df["Close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["ema_fast_slope"] = _slope(df["ema_fast"], presets["ema_slope_lb"]).fillna(0.0)

    # ---- HTF filter ----
    use_htf   = bool(filters.get("use_htf", True))
    htf_rule  = filters.get("htf_rule", presets["htf_rule"])
    htf_ema_n = int(filters.get("htf_ema_len", 20))
    if use_htf:
        df["htf_ok"] = _resample_htf(df, htf_rule, htf_ema_n)
    else:
        df["htf_ok"] = True

    # ---- Quality gates ----
    adx_min = max(presets["adx_min"], int(filters.get("adx_min", presets["adx_min"])))
    min_atr_pct = max(presets["min_atr_pct"], float(filters.get("min_atr_pct", presets["min_atr_pct"])))

    df["quality_ok"] = (
        (df["adx"] >= adx_min)
        & (df["atr_pct"] >= min_atr_pct)
        & (df["ema_fast_slope"] > 0)
        & (df["htf_ok"])
    )

    # ---- Entry / Exit (long-only for now) ----
    rsi_buy  = max(presets["rsi_buy"], int(b.get("rsi_buy", cfg.get("rsi_buy", presets["rsi_buy"]))))
    rsi_sell = min(presets["rsi_sell"], int(b.get("rsi_sell", cfg.get("rsi_sell", presets["rsi_sell"]))))

    df["enter_long"] = (
        (df["Close"] > df["ema_fast"] * (1 + ema_poke))
        & (df["ema_fast"] > df["ema_slow"])
        & (df["rsi"] >= rsi_buy)
        & (df["quality_ok"])
    ).fillna(False)

    df["exit_long_hint"] = ((df["rsi"] <= rsi_sell) | (df["Close"] < df["ema_fast"])).fillna(False)

    # placeholders for symmetry
    df["enter_short"] = False
    df["exit_short_hint"] = False

    # ---- Risk-based position sizing (safe int cast) ----
    capital    = float(cfg.get("capital_rs", 100_000.0))
    risk_perc  = float(cfg.get("risk_perc", 0.01))  # 1% risk default

    atr = df["atr"].values
    pos_size_float = np.where(atr > 0.0, (capital * risk_perc / atr), 0.0)
    # clamp to avoid absurd sizes, then convert to int safely
    df["pos_size"] = np.clip(pos_size_float, 0, 1_000_000).astype(np.int64)

    return df
