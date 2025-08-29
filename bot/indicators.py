# bot/indicators.py
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Adds standard indicators with column names that the strategy expects.
    Expects OHLC columns: Open, High, Low, Close
    """
    d = df.copy()

    # windows from cfg (with sane defaults)
    ema_fast_len = int(cfg.get("ema_fast", 21))
    ema_slow_len = int(cfg.get("ema_slow", 50))
    rsi_len      = int(cfg.get("rsi_len", 14))
    adx_len      = int(cfg.get("adx_len", 14))
    atr_len      = int(cfg.get("atr_len", 14))

    px = d["Close"]
    d["ema_fast"] = EMAIndicator(px, window=ema_fast_len).ema_indicator()
    d["ema_slow"] = EMAIndicator(px, window=ema_slow_len).ema_indicator()
    d["rsi"]      = RSIIndicator(px, window=rsi_len).rsi()
    d["adx"]      = ADXIndicator(d["High"], d["Low"], px, window=adx_len).adx()
    d["atr"]      = AverageTrueRange(d["High"], d["Low"], px, window=atr_len).average_true_range()

    # drop warmup NAs
    return d.dropna().reset_index(drop=True)
