# bot/strategy.py
from __future__ import annotations
import pandas as pd
import numpy as np

from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# ---------------------------
# Small TA helpers
# ---------------------------
def _ema(series: pd.Series, n: int) -> pd.Series:
    return EMAIndicator(close=series, window=n).ema_indicator()

def _rsi(series: pd.Series, n: int) -> pd.Series:
    return RSIIndicator(close=series, window=n).rsi()

def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    return ADXIndicator(high=h, low=l, close=c, window=n).adx()

def _atr(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    return AverageTrueRange(high=h, low=l, close=c, window=n).average_true_range()


def _resample_htf(df: pd.DataFrame, rule: str, ema_len: int, slope_lb: int = 3) -> pd.Series:
    """
    15m/5m જેવી higher timeframe પર EMA trend filter.
    """
    ohlc = df[['Open','High','Low','Close']].resample(rule, label="right", closed="right").agg(
        {'Open':'first','High':'max','Low':'min','Close':'last'}
    ).dropna()
    ema = _ema(ohlc['Close'], ema_len)
    slope = ema.diff(slope_lb)
    htf_ok = ((ohlc['Close'] > ema) & (slope > 0)).reindex(df.index, method="ffill").fillna(False)
    return htf_ok.astype(bool)


def _slope(x: pd.Series, lookback: int = 5) -> pd.Series:
    return x.diff(lookback)


# ---------------------------
# Profiles / Presets
# ---------------------------
def profile_presets(profile: str) -> dict:
    """
    Presets per strictness:
      - ADX minimum
      - RSI buy/sell
      - EMA slope lookback
      - ATR% min/max window
      - HTF rule + HTF EMA len
      - Entry poke over EMA
    """
    p = (profile or "loose").lower()
    if p == "strict":
        return dict(
            adx_min=18, rsi_buy=56, rsi_sell=45,
            ema_slope_lb=5, min_atr_pct=0.00035, max_atr_pct=0.01,
            htf_rule="15min", htf_ema=20, ema_poke_pct=0.0008
        )
    if p == "medium":
        return dict(
            adx_min=16, rsi_buy=54, rsi_sell=46,
            ema_slope_lb=4, min_atr_pct=0.00030, max_atr_pct=0.012,
            htf_rule="15min", htf_ema=20, ema_poke_pct=0.0006
        )
    # loose (default)
    return dict(
        adx_min=14, rsi_buy=52, rsi_sell=47,
        ema_slope_lb=3, min_atr_pct=0.00025, max_atr_pct=0.013,
        htf_rule="15min", htf_ema=20, ema_poke_pct=0.0005
    )


# ---------------------------
# Main signal builder
# ---------------------------
def prepare_signals(prices: pd.DataFrame, cfg: dict, profile: str = "loose") -> pd.DataFrame:
    """
    Returns df with at least: enter_long (bool), atr (float).
    Engine (bot/backtest.py) handles SL/TP exits with ATR multiples.
    """
    df = prices.copy().sort_index()

    presets = profile_presets(profile)

    # read overrides from cfg if provided
    bcfg = (cfg.get("backtest") or {})
    ema_fast = int(bcfg.get("ema_fast", cfg.get("ema_fast", 21)))
    ema_slow = int(bcfg.get("ema_slow", cfg.get("ema_slow", 50)))
    rsi_len  = int(bcfg.get("rsi_len", 14))
    adx_len  = int((bcfg.get("filters", {}) or {}).get("adx_len", 14))
    atr_len  = int(bcfg.get("atr_len", cfg.get("atr_len", 14)))
    ema_poke = float(bcfg.get("ema_poke_pct", presets["ema_poke_pct"]))

    # core indicators
    df["ema_fast"] = _ema(df["Close"], ema_fast)
    df["ema_slow"] = _ema(df["Close"], ema_slow)
    df["rsi"]      = _rsi(df["Close"], rsi_len)
    df["adx"]      = _adx(df["High"], df["Low"], df["Close"], adx_len)
    df["atr"]      = _atr(df["High"], df["Low"], df["Close"], atr_len)

    # volatility as % of price
    with np.errstate(divide="ignore", invalid="ignore"):
        df["atr_pct"] = (df["atr"] / df["Close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # slope (trend persistence)
    df["ema_fast_slope"] = _slope(df["ema_fast"], presets["ema_slope_lb"]).fillna(0.0)

    # HTF filter
    use_htf = bool((bcfg.get("filters", {}) or {}).get("use_htf", True))
    htf_rule = (bcfg.get("filters", {}) or {}).get("htf_rule", presets["htf_rule"])
    htf_ema_len = int((bcfg.get("filters", {}) or {}).get("htf_ema_len", presets["htf_ema"]))
    if use_htf:
        df["htf_ok"] = _resample_htf(df, htf_rule, htf_ema_len, slope_lb=presets["ema_slope_lb"])
    else:
        df["htf_ok"] = True

    # quality gate
    min_atr = float((bcfg.get("filters", {}) or {}).get("min_atr_pct", presets["min_atr_pct"]))
    max_atr = float((bcfg.get("filters", {}) or {}).get("max_atr_pct", presets["max_atr_pct"]))
    adx_min = int((bcfg.get("filters", {}) or {}).get("adx_min", presets["adx_min"]))

    df["quality_ok"] = (
        (df["adx"] >= adx_min) &
        (df["atr_pct"] >= min_atr) &
        (df["atr_pct"] <= max_atr) &
        (df["ema_fast_slope"] > 0) &
        (df["htf_ok"])
    )

    # thresholds
    rsi_buy  = int(bcfg.get("rsi_buy",  presets["rsi_buy"]))
    rsi_sell = int(bcfg.get("rsi_sell", presets["rsi_sell"]))

    # entries (long-only)
    df["enter_long"] = (
        (df["Close"] > df["ema_fast"] * (1.0 + ema_poke)) &
        (df["ema_fast"] > df["ema_slow"]) &
        (df["rsi"] >= rsi_buy) &
        (df["quality_ok"])
    )

    # optional exit hint (not required by engine but useful for future)
    df["exit_long_hint"] = (df["rsi"] <= rsi_sell) | (df["Close"] < df["ema_fast"])

    # disable shorts for now
    df["enter_short"] = False
    df["exit_short_hint"] = False

    # Clean unsafe numeric artefacts
    for col in ["ema_fast","ema_slow","rsi","adx","atr","atr_pct","ema_fast_slope"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df
