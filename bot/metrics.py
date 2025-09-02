# bot/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _duration_minutes_safe(r) -> float:
    """Return holding time in minutes; NaN for open/invalid rows."""
    et = r.get("exit_time", pd.NaT)
    st = r.get("entry_time", pd.NaT)
    if pd.isna(et) or pd.isna(st):
        return np.nan
    if not isinstance(et, pd.Timestamp) or not isinstance(st, pd.Timestamp):
        # try to coerce common types
        try:
            et = pd.to_datetime(et, errors="coerce")
            st = pd.to_datetime(st, errors="coerce")
        except Exception:
            return np.nan
        if pd.isna(et) or pd.isna(st):
            return np.nan
    delta = et - st
    # some libs may return numpy timedelta64
    try:
        return float(delta.total_seconds()) / 60.0
    except AttributeError:
        # pandas Timedelta supports .total_seconds() too
        try:
            return float(pd.to_timedelta(delta).total_seconds()) / 60.0
        except Exception:
            return np.nan

def compute_metrics(trades: pd.DataFrame, equity: pd.Series, starting_capital: float):
    """
    Numeric evaluation for backtests/live runs.
    Returns a flat dict safe to JSON.
    """
    # ROI / drawdown / time in drawdown
    if equity is None or equity.empty:
        roi = 0.0
        max_dd_pct = 0.0
        time_dd = 0
    else:
        roi = (equity.iloc[-1] - starting_capital) / starting_capital * 100.0
        roll_max = equity.cummax()
        dd = equity / roll_max - 1.0
        max_dd_pct = float(dd.min() * 100.0)
        time_dd = 0
        cur = 0
        for below in (equity < roll_max):
            cur = cur + 1 if below else 0
            time_dd = max(time_dd, cur)

    # trade stats
    n_trades = int(len(trades)) if trades is not None else 0
    if trades is None or trades.empty:
        win_rate = 0.0
        rr = 0.0
        sharpe = 0.0
        profit_factor = 0.0
        expectancy = 0.0
        avg_hold_minutes = 0.0
    else:
        pnl = trades["pnl"].to_numpy()
        wins = pnl[pnl > 0]
        losses = -pnl[pnl < 0]

        win_rate = float(round((len(wins) / n_trades) * 100.0, 2)) if n_trades else 0.0
        avg_win = float(np.mean(wins)) if len(wins) else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) else np.nan
        rr = float(round((avg_win / avg_loss), 2)) if (np.isfinite(avg_loss) and avg_loss > 0) else 0.0

        gross_profit = float(np.sum(wins)) if len(wins) else 0.0
        gross_loss = float(np.sum(losses)) if len(losses) else 0.0
        profit_factor = float(round(gross_profit / gross_loss, 2)) if gross_loss > 0 else 0.0

        p = (len(wins) / n_trades) if n_trades else 0.0
        q = 1.0 - p
        expectancy = float(round(p * avg_win - q * (avg_loss if np.isfinite(avg_loss) else 0.0), 2))

        # Sharpe (bar returns) — guard std=0
        if equity is not None and not equity.empty:
            returns = equity.pct_change().dropna()
            if not returns.empty and returns.std() > 0:
                sharpe = float(round((returns.mean() / returns.std() * np.sqrt(252)), 2))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # average holding time (minutes) — robust to NaT/open trades
        if "entry_time" in trades and "exit_time" in trades:
            durations = trades.apply(_duration_minutes_safe, axis=1)
            avg_hold_minutes = float(round(np.nanmean(durations) if np.isfinite(np.nanmean(durations)) else 0.0, 2))
        else:
            avg_hold_minutes = 0.0

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "roi_pct": float(round(roi, 2)),
        "max_dd_pct": float(round(max_dd_pct, 2)),
        "time_dd_bars": int(time_dd),
        "rr": rr,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_hold_minutes": avg_hold_minutes,
        "sharpe_ratio": sharpe,
    }
