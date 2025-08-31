# bot/strategy.py
from __future__ import annotations
import numpy as np
import pandas as pd

from .indicators import add_indicators  # must add ATR + VWAP (or we calc VWAP here)

LONG, SHORT, FLAT = 1, -1, 0


def _intraday_session_key(ts: pd.Timestamp) -> tuple:
    d = ts.tz_localize(None) if ts.tzinfo else ts
    return (d.date(),)  # simple day splitter


def _compute_orb(df: pd.DataFrame, first_minutes: int) -> pd.DataFrame:
    """
    For each trading day, compute the ORB high/low from first N minutes bars.
    Assumes df.index is a DatetimeIndex (IST) and has High/Low.
    """
    d = df.copy()
    day_keys = d.index.map(_intraday_session_key)
    d["day_key"] = day_keys

    orb_hi = []
    orb_lo = []
    seen = set()

    for k in d["day_key"].unique():
        day_mask = d["day_key"] == k
        day_df = d.loc[day_mask]

        # select bars within first N minutes of that day
        day_start = day_df.index[0]
        cutoff = day_start + pd.Timedelta(minutes=first_minutes)
        orb_window = day_df.loc[(day_df.index >= day_start) & (day_df.index < cutoff)]

        hi = float(orb_window["High"].max()) if not orb_window.empty else np.nan
        lo = float(orb_window["Low"].min())  if not orb_window.empty else np.nan

        orb_hi.append(pd.Series(hi, index=day_df.index))
        orb_lo.append(pd.Series(lo, index=day_df.index))

    d["orb_high"] = pd.concat(orb_hi).sort_index()
    d["orb_low"]  = pd.concat(orb_lo).sort_index()
    return d.drop(columns=["day_key"])


def _compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Simple per-day VWAP using typical price and volume.
    """
    d = df.copy()
    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    vol = d.get("Volume", pd.Series(index=d.index, data=1.0))
    day = d.index.normalize()

    # cumulative within day
    num = (tp * vol).groupby(day).cumsum()
    den = vol.groupby(day).cumsum()
    vwap = num / den.replace(0, np.nan)
    return vwap


def prepare_signals(prices: pd.DataFrame, cfg: dict, use_block: str = "intraday_options") -> pd.DataFrame:
    """
    Build dataframe with:
      - ORB high/low (first N minutes)
      - VWAP
      - Entry signals (long/short)
      - 'signal' column in {1, -1, 0}
    Expected columns: ['Open','High','Low','Close','Volume?'] with DatetimeIndex (IST).
    """
    block = (cfg.get(use_block) or {})
    entry_cfg = block.get("entry", {})
    exits_cfg = block.get("exits", {})

    d = prices.copy()
    d = add_indicators(d, {  # ensure ATR available; EMA/ADX optional
        "atr_len": exits_cfg.get("atr_len", 14) if "atr_len" in exits_cfg else 14
    })

    # ORB
    first_min = int(entry_cfg.get("orb_first_minutes", 15))
    d = _compute_orb(d, first_min)

    # VWAP
    d["vwap"] = _compute_intraday_vwap(d)

    # ENTRY logic
    use_vwap = bool(entry_cfg.get("use_vwap_filter", True))
    vwap_side = str(entry_cfg.get("vwap_side", "with_trend")).lower()

    # breakout conditions
    long_ok = (d["Close"] > d["orb_high"])
    short_ok = (d["Close"] < d["orb_low"])

    if use_vwap:
        if vwap_side == "with_trend":
            long_ok = long_ok & (d["Close"] >= d["vwap"])
            short_ok = short_ok & (d["Close"] <= d["vwap"])
        elif vwap_side == "both":
            # allow both-sided trades but still reference VWAP to avoid whipsaws
            pass
        else:
            # unknown -> default to with_trend
            long_ok = long_ok & (d["Close"] >= d["vwap"])
            short_ok = short_ok & (d["Close"] <= d["vwap"])

    d["long_entry"]  = long_ok.fillna(False)
    d["short_entry"] = short_ok.fillna(False)

    d["signal"] = 0
    d.loc[d["long_entry"], "signal"] = LONG
    d.loc[d["short_entry"], "signal"] = SHORT
    return d


def initial_stop_target(side: int, entry_price: float, atr: float, exits_cfg: dict):
    """
    Stop = entry ± ATR * sl_atr_mult
    Target = entry ± RR * (stop distance)
    """
    sl_mult = float(exits_cfg.get("sl_atr_mult", 1.0))
    rr      = float(exits_cfg.get("tgt_rr", 1.5))
    stop_dist = sl_mult * atr

    if side == LONG:
        stop   = entry_price - stop_dist
        target = entry_price + rr * stop_dist
    else:
        stop   = entry_price + stop_dist
        target = entry_price - rr * stop_dist
    return float(stop), float(target)


def trail_stop(side: int, price: float, atr: float, current_stop: float, entry: float, exits_cfg: dict):
    """
    ATR trailing:
      - Start after unrealized >= trail_start_rr * R (R = ATR*sl_mult)
      - Then trail by trail_atr_mult * ATR
    """
    trail_start_rr  = float(exits_cfg.get("trail_start_rr", 1.0))
    trail_atr_mult  = float(exits_cfg.get("trail_atr_mult", 1.0))
    sl_mult         = float(exits_cfg.get("sl_atr_mult", 1.0))
    trigger_points  = trail_start_rr * sl_mult * atr
    trail_dist      = trail_atr_mult  * atr

    if side == LONG:
        if (price - entry) >= trigger_points:
            new_stop = price - trail_dist
            return max(current_stop, new_stop)
        return current_stop
    else:
        if (entry - price) >= trigger_points:
            new_stop = price + trail_dist
            return min(current_stop, new_stop)
        return current_stop
