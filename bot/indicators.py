import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = df.copy()
    px = d["Close"]
    d["ema_f"] = EMAIndicator(px, window=cfg["ema_fast"]).ema_indicator()
    d["ema_s"] = EMAIndicator(px, window=cfg["ema_slow"]).ema_indicator()
    d["rsi"]   = RSIIndicator(px, window=cfg["rsi_len"]).rsi()
    d["adx"]   = ADXIndicator(d["High"], d["Low"], px, window=cfg["adx_len"]).adx()
    d["atr"]   = AverageTrueRange(d["High"], d["Low"], px, window=cfg["atr_len"]).average_true_range()
    return d.dropna().reset_index(drop=True)
