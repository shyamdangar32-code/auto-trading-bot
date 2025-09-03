# bot/strategy.py
from __future__ import annotations
import pandas as pd
import numpy as np

# indicators from 'ta' (already in requirements)
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def _ema(series: pd.Series, n: int) -> pd.Series:
    return EMAIndicator(close=series, window=n).ema_indicator()

def _rsi(series: pd.Series, n: int) -> pd.Series:
    return RSIIndicator(close=series, window=n).rsi()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return ADXIndicator(high=high, low=low, close=close, window=n).adx()

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return AverageTrueRange(high=high, low=low, close=close, window=n).average_true_range()


def _resample_htf(df: pd.DataFrame, rule: str, ema_len: int) -> pd.Series:
    """
    Higher-timeframe confirmation: price above HTF EMA.
    Returns a Series aligned to df.index with boolean confirmation.
    """
    ohlc = df[['Open','High','Low','Close']].resample(rule, label="right", closed="right").agg({
        'Open':'first','High':'max','Low':'min','Close':'last'
    }).dropna()
    htf_ema = _ema(ohlc['Close'], ema_len)
    htf_ok = (ohlc['Close'] > htf_ema).reindex(df.index, method="ffill").fillna(False)
    return htf_ok.astype(bool)


def _slope(x: pd.Series, lookback: int = 5) -> pd.Series:
    """simple slope of EMA — positive means trending up"""
    return x.diff(lookback)


def profile_presets(profile: str) -> dict:
    """
    Thresholds per profile. 'loose' = more trades, 'strict' = fewer & cleaner.
    """
    p = (profile or "loose").lower()
    if p == "strict":
        return dict(adx_min=20, rsi_buy=55, rsi_sell=45, ema_slope_lb=5, min_atr_pct=0.0004, htf_rule="15min")
    if p == "medium":
        return dict(adx_min=16, rsi_buy=53, rsi_sell=47, ema_slope_lb=4, min_atr_pct=0.00035, htf_rule="15min")
    # loose
    return dict(adx_min=12, rsi_buy=52, rsi_sell=48, ema_slope_lb=3, min_atr_pct=0.00030, htf_rule="15min")


def prepare_signals(prices: pd.DataFrame, cfg: dict, profile: str = "loose") -> pd.DataFrame:
    """
    Adds indicator columns + entry/exit boolean columns.
    Signals are LONG-only (index/options bias), but SHORT flags included for future.
    """
    df = prices.copy().sort_index()
    presets = profile_presets(profile)

    # base params (mirrors config)
    ema_fast = int((cfg.get("backtest") or {}).get("ema_fast", cfg.get("ema_fast", 21)))
    ema_slow = int((cfg.get("backtest") or {}).get("ema_slow", cfg.get("ema_slow", 50)))
    rsi_len  = int((cfg.get("backtest") or {}).get("rsi_len", 14))
    adx_len  = int(((cfg.get("backtest") or {}).get("filters") or {}).get("adx_len", 14))
    atr_len  = int((cfg.get("backtest") or {}).get("atr_len", cfg.get("atr_len", 14)))
    ema_poke = float((cfg.get("backtest") or {}).get("ema_poke_pct", cfg.get("ema_poke_pct", 0.0001)))

    # indicators
    df["ema_fast"] = _ema(df["Close"], ema_fast)
    df["ema_slow"] = _ema(df["Close"], ema_slow)
    df["rsi"]      = _rsi(df["Close"], rsi_len)
    df["adx"]      = _adx(df["High"], df["Low"], df["Close"], adx_len)
    df["atr"]      = _atr(df["High"], df["Low"], df["Close"], atr_len)
    df["atr_pct"]  = (df["atr"] / df["Close"]).fillna(0.0)

    df["ema_fast_slope"] = _slope(df["ema_fast"], presets["ema_slope_lb"]).fillna(0.0)

    # HTF confirmation
    use_htf = bool(((cfg.get("backtest") or {}).get("filters") or {}).get("use_htf", True))
    htf_rule = ((cfg.get("backtest") or {}).get("filters") or {}).get("htf_rule", presets["htf_rule"])
    htf_ema_len = int(((cfg.get("backtest") or {}).get("filters") or {}).get("htf_ema_len", 20))
    if use_htf:
        df["htf_ok"] = _resample_htf(df, htf_rule, htf_ema_len)
    else:
        df["htf_ok"] = True

    # quality filters
    df["quality_ok"] = (
        (df["adx"] >= max(presets["adx_min"], int(((cfg.get("backtest") or {}).get("filters") or {}).get("adx_min", presets["adx_min"]))))
        & (df["atr_pct"] >= max(presets["min_atr_pct"], float(((cfg.get("backtest") or {}).get("filters") or {}).get("min_atr_pct", presets["min_atr_pct"]))))
        & (df["ema_fast_slope"] > 0)
        & (df["htf_ok"])
    )

    # entry logic (LONG)
    rsi_buy = max(presets["rsi_buy"], int((cfg.get("backtest") or {}).get("rsi_buy", cfg.get("rsi_buy", presets["rsi_buy"]))))
    rsi_sell = min(presets["rsi_sell"], int((cfg.get("backtest") or {}).get("rsi_sell", cfg.get("rsi_sell", presets["rsi_sell"]))))

    df["enter_long"] = (
        (df["Close"] > df["ema_fast"] * (1 + ema_poke))
        & (df["ema_fast"] > df["ema_slow"])
        & (df["rsi"] >= rsi_buy)
        & (df["quality_ok"])
    )

    # simple long exit hint (unused by engine when using SL/TP)
    df["exit_long_hint"] = (df["rsi"] <= rsi_sell) | (df["Close"] < df["ema_fast"])

    # SHORT hooks for future
    df["enter_short"] = False
    df["exit_short_hint"] = False

    return df
