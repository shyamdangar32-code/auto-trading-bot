# bot/strategy.py  (UNIFIED + ENTRY WINDOWS GATE)
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

# --- indicators (fallbacks if local ones not available) ---
try:
    from .indicators import ema, rsi, atr, adx  # type: ignore
except Exception:
    def ema(s: pd.Series, length: int) -> pd.Series:
        return s.ewm(span=int(length), adjust=False).mean()

    def rsi(close: pd.Series, length: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(length).mean()
        loss = -delta.where(delta < 0, 0.0).rolling(length).mean().replace(0, np.nan)
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(length).mean()

    def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        up = high.diff()
        dn = -low.diff()
        plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
        minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
        trur = atr(high, low, close, length)
        plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(length).mean() / trur
        minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(length).mean() / trur
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
        return dx.rolling(length).mean()

LONG, SHORT, FLAT = +1, -1, 0

@dataclass
class Plan:
    ema_fast: int = 21
    ema_slow: int = 50
    rsi_len: int = 14
    rsi_buy: float = 52.0
    rsi_sell: float = 48.0
    adx_len: int = 14
    atr_len: int = 14
    atr_mult_sl: float = 1.8
    atr_mult_tp: float = 2.5
    risk_perc: float = 0.002
    allow_shorts: bool = False
    trend_filter: bool = True
    htf_minutes: int = 5
    order_qty: int = 1
    capital_rs: float = 100000.0
    strategy: str = "ema_rsi_adx"

def _apply_entry_windows(out: pd.DataFrame, plan: Dict) -> None:
    """Zero out entry signals outside allowed time windows."""
    import datetime as _dt
    wins_cfg = plan.get("entry_windows") or []
    if not wins_cfg:
        return
    def _win_to_times(s: str):
        a, b = s.split("-", 1)
        return _dt.time.fromisoformat(a), _dt.time.fromisoformat(b)
    wins = [_win_to_times(w) for w in wins_cfg]
    t = out.index.time
    allow = np.zeros(len(out), dtype=bool)
    for (st, en) in wins:
        allow |= ((t >= st) & (t <= en))
    out["signal"] = np.where(allow, out["signal"], 0)

def _intraday_rsi_signals(df: pd.DataFrame, plan: Dict) -> pd.DataFrame:
    out = df.copy()
    for c in ("Open", "High", "Low", "Close"):
        if c in out.columns:
            out[c.lower()] = out[c]
    close = out["close"]; high = out["high"]; low = out["low"]

    rlen = int(plan.get("rsi_len", 14))
    ob   = float(plan.get("rsi_overbought", 70))
    os   = float(plan.get("rsi_oversold", 30))
    out["rsi"] = rsi(close, rlen)

    sig = np.where(out["rsi"] < os, LONG, np.where(out["rsi"] > ob, SHORT, FLAT))

    if plan.get("trend_filter", True):
        m = int(plan.get("htf_minutes", 5))
        htf = out.resample(f"{m}T").last()
        htf["ema_f"] = ema(htf["close"], int(plan.get("ema_fast", 21)))
        htf["ema_s"] = ema(htf["close"], int(plan.get("ema_slow", 50)))
        htf["up"] = htf["ema_f"] > htf["ema_s"]
        htf["dn"] = htf["ema_f"] < htf["ema_s"]
        out[["up","dn"]] = htf[["up","dn"]].reindex(out.index).ffill()
        sig = np.where((sig == LONG) & (out["up"]), LONG,
              np.where((sig == SHORT) & (out["dn"]), SHORT, FLAT))

    if not plan.get("allow_shorts", False):
        sig = np.where(sig == SHORT, FLAT, sig)

    out["signal"] = sig
    _apply_entry_windows(out, plan)

    atr_len = int(plan.get("atr_len", 14))
    out["atr"] = atr(high, low, close, atr_len)
    sl_mult = float(plan.get("atr_mult_sl", 1.8))
    tp_mult = float(plan.get("atr_mult_tp", 2.5))
    entry_ref = close

    long_sl  = entry_ref - sl_mult*out["atr"]
    long_tp  = entry_ref + tp_mult*out["atr"]
    short_sl = entry_ref + sl_mult*out["atr"]
    short_tp = entry_ref - tp_mult*out["atr"]

    out["sl_px"] = np.where(out["signal"]==LONG, long_sl,
                     np.where(out["signal"]==SHORT, short_sl, np.nan))
    out["tp_px"] = np.where(out["signal"]==LONG, long_tp,
                     np.where(out["signal"]==SHORT, short_tp, np.nan))

    fallback_qty = int(plan.get("order_qty", 1))
    risk_rs = float(plan.get("capital_rs", 100000)) * float(plan.get("risk_perc", 0.002))
    dist = (entry_ref - out["sl_px"]).abs()
    out["pos_size"] = np.where((dist>0) & np.isfinite(dist), np.maximum((risk_rs/dist).round(), 1), fallback_qty)

    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    out.dropna(subset=["signal","sl_px","tp_px","atr"], inplace=True)
    return out

def _ema_rsi_adx_signals(df: pd.DataFrame, plan: Dict) -> pd.DataFrame:
    out = df.copy()
    for c in ("Open", "High", "Low", "Close"):
        if c in out.columns:
            out[c.lower()] = out[c]
    close = out["close"]; high = out["high"]; low = out["low"]

    ef = int(plan.get("ema_fast", 21))
    es = int(plan.get("ema_slow", 50))
    rl = int(plan.get("rsi_len", 14))
    al = int(plan.get("adx_len", 14))
    atr_len = int(plan.get("atr_len", 14))

    out["ema_f"] = ema(close, ef)
    out["ema_s"] = ema(close, es)
    out["rsi"]   = rsi(close, rl)
    out["atr"]   = atr(high, low, close, atr_len)
    out["adx"]   = adx(high, low, close, al)

    sig_long  = (out["ema_f"] > out["ema_s"]) & (out["rsi"] > float(plan.get("rsi_buy", 52.0)))
    sig_short = (out["ema_f"] < out["ema_s"]) & (out["rsi"] < float(plan.get("rsi_sell", 48.0)))
    sig = np.where(sig_long, LONG, np.where(sig_short, SHORT, FLAT))

    if not plan.get("allow_shorts", False):
        sig = np.where(sig == SHORT, FLAT, sig)

    out["signal"] = sig
    _apply_entry_windows(out, plan)

    sl_mult = float(plan.get("atr_mult_sl", 1.8))
    tp_mult = float(plan.get("atr_mult_tp", 2.5))
    entry_ref = close

    long_sl  = entry_ref - sl_mult*out["atr"]
    long_tp  = entry_ref + tp_mult*out["atr"]
    short_sl = entry_ref + sl_mult*out["atr"]
    short_tp = entry_ref - tp_mult*out["atr"]

    out["sl_px"] = np.where(out["signal"]==LONG, long_sl,
                     np.where(out["signal"]==SHORT, short_sl, np.nan))
    out["tp_px"] = np.where(out["signal"]==LONG, long_tp,
                     np.where(out["signal"]==SHORT, short_tp, np.nan))

    fallback_qty = int(plan.get("order_qty", 1))
    risk_rs = float(plan.get("capital_rs", 100000)) * float(plan.get("risk_perc", 0.002))
    dist = (entry_ref - out["sl_px"]).abs()
    out["pos_size"] = np.where((dist>0) & np.isfinite(dist), np.maximum((risk_rs/dist).round(), 1), fallback_qty)

    out.replace([np.inf,-np.inf], np.nan, inplace=True)
    out.dropna(subset=["signal","sl_px","tp_px","atr"], inplace=True)
    return out

def prepare_signals(prices: pd.DataFrame, plan: Dict) -> pd.DataFrame:
    name = str(plan.get("strategy", "ema_rsi_adx")).lower()
    if name in ("intraday_rsi", "rsi_simple"):
        return _intraday_rsi_signals(prices, plan)
    return _ema_rsi_adx_signals(prices, plan)
