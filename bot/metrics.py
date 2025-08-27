# bot/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

def compute_metrics(trades: pd.DataFrame, equity: pd.Series, starting_capital: float):
    """
    Numeric evaluation for backtests/live runs.
    Returns a flat dict safe to JSON.
    """
    # ROI / drawdown / time in dd
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
        rr = float(round((avg_win / avg_loss), 2)) if avg_loss and not np.isnan(avg_loss) and avg_loss > 0 else 0.0

        gross_profit = float(np.sum(wins)) if len(wins) else 0.0
        gross_loss = float(np.sum(losses)) if len(losses) else 0.0
        profit_factor = float(round(gross_profit / gross_loss, 2)) if gross_loss > 0 else 0.0

        p = (len(wins) / n_trades) if n_trades else 0.0
        q = 1.0 - p
        expectancy = float(round(p * avg_win - q * (avg_loss if np.isfinite(avg_loss) else 0.0), 2))

        # Sharpe (bar returns) — conservative approximation
        returns = equity.pct_change().dropna() if equity is not None else pd.Series(dtype=float)
        sharpe = float(round((returns.mean() / returns.std() * np.sqrt(252)) if not returns.empty else 0.0, 2))

        # average holding time (minutes)
        avg_hold_minutes = float(round(
            trades.apply(lambda r: (r["exit_time"] - r["entry_time"]).total_seconds() / 60, axis=1).mean()
            , 2)) if ("entry_time" in trades and "exit_time" in trades and not trades.empty) else 0.0

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
