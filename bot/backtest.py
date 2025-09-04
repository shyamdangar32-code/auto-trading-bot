# bot/backtest.py
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, List
from datetime import time as dtime

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_hhmm(s: str) -> dtime:
    """'HH:MM' → datetime.time"""
    sh, sm = map(int, s.strip().split(":"))
    return dtime(hour=sh, minute=sm)

def _to_time(x) -> dtime:
    """Any ts → datetime.time (robust for pd.Timestamp/np.datetime64/str)"""
    if isinstance(x, dtime):
        return x
    try:
        return pd.Timestamp(x).time()
    except Exception:
        # last resort (e.g. '09:20')
        return _parse_hhmm(str(x))

def _within_session(ts, sess_start, sess_end) -> bool:
    """
    Compare only times (no dates). Avoid pd.Timestamp(hour=..) which needs 'year'.
    """
    t  = _to_time(ts)
    s  = _parse_hhmm(sess_start) if isinstance(sess_start, str) else _to_time(sess_start)
    e  = _parse_hhmm(sess_end)   if isinstance(sess_end, str)   else _to_time(sess_end)

    # handle normal daytime window (e.g., 09:20–15:20)
    if s <= e:
        return (t >= s) and (t <= e)
    # if a window crosses midnight (not our case, but safe)
    return (t >= s) or (t <= e)

# ──────────────────────────────────────────────────────────────────────────────
# Simple backtest engine (same signature tools/run_backtest.py expects)
# Returns: (summary_dict, trades_df, equity_series)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_px: float
    exit_px: float
    pnl_rs: float

def run_backtest(
    prices: pd.DataFrame,
    cfg: Dict[str, Any],
    use_block: str | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series]:
    """
    Very lightweight long-only engine. Keeps public signature stable.
    """
    bt = (cfg.get("backtest") or {})
    sess_s = bt.get("session_start", "09:20")
    sess_e = bt.get("session_end",   "15:20")

    # indicators / signals are prepared outside and attached on prices
    df = prices.copy().sort_index()

    # keep only session bars
    mask = df.index.map(lambda ts: _within_session(ts, sess_s, sess_e))
    df = df.loc[mask].copy()

    # fallbacks if signals absent
    enter = df.get("enter_long", pd.Series(False, index=df.index))
    exit_hint = df.get("exit_long_hint", pd.Series(False, index=df.index))

    # trading params (minimal; brokerage/slippage handled elsewhere)
    order_qty = int(cfg.get("order_qty", 1))
    capital   = float(cfg.get("capital_rs", 100_000))

    in_pos = False
    entry_px = np.nan
    entry_ts = pd.NaT
    trades: List[Trade] = []
    equity = [capital]
    last_ts = None

    for ts, row in df.iterrows():
        last_ts = ts
        px = float(row["Close"])

        if not in_pos and bool(enter.loc[ts]):
            # enter
            in_pos = True
            entry_px = px
            entry_ts = ts
        elif in_pos and bool(exit_hint.loc[ts]):
            # exit
            pnl = (px - entry_px) * order_qty
            trades.append(Trade(entry_ts, ts, entry_px, px, pnl))
            capital += pnl
            in_pos = False
            entry_px = np.nan
            entry_ts = pd.NaT

        equity.append(capital)

    # force close open position at last bar
    if in_pos and last_ts is not None:
        px = float(df.loc[last_ts, "Close"])
        pnl = (px - entry_px) * order_qty
        trades.append(Trade(entry_ts, last_ts, entry_px, px, pnl))
        capital += pnl
        equity.append(capital)

    equity_ser = pd.Series(equity, index=range(len(equity)), name="equity_rs")

    # summary
    pnl_list = [t.pnl_rs for t in trades]
    total_trades = len(trades)
    wins = sum(1 for p in pnl_list if p > 0)
    losses = sum(1 for p in pnl_list if p < 0)
    gross_profit = sum(p for p in pnl_list if p > 0)
    gross_loss   = -sum(p for p in pnl_list if p < 0)

    pf = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
    roi_pct = ((equity_ser.iloc[-1] - equity_ser.iloc[0]) / max(1.0, equity_ser.iloc[0])) * 100.0

    max_dd = 0.0
    peak = -np.inf
    for v in equity_ser:
        if v > peak:
            peak = v
        dd = (v - peak)
        if dd < max_dd:
            max_dd = dd
    max_dd_pct = (max_dd / equity_ser.iloc[0]) * 100.0 if equity_ser.iloc[0] else 0.0

    summary = {
        "trades": total_trades,
        "win_rate_pct": (wins / total_trades * 100.0) if total_trades else 0.0,
        "roi_pct": roi_pct,
        "pf": pf if np.isfinite(pf) else 0.0,
        "max_dd_pct": max_dd_pct,
    }

    trades_df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame(
        columns=["entry_time","exit_time","entry_px","exit_px","pnl_rs"]
    )
    return summary, trades_df, equity_ser
