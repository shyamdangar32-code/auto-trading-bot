# bot/metrics.py
import numpy as np
import pandas as pd

def compute_metrics(trades: pd.DataFrame, equity: pd.Series, starting_capital: float):
    """
    Compute extended metrics for both backtests and live runs.
    """
    if equity is None or equity.empty:
        return {"n_trades": 0, "win_rate": 0.0, "roi_pct": 0.0}

    # ROI
    roi = (equity.iloc[-1] - starting_capital) / starting_capital * 100.0

    # Drawdown
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = dd.min() * 100.0

    # Time in drawdown
    time_dd = 0
    curr = 0
    for below in (equity < roll_max):
        curr = curr + 1 if below else 0
        time_dd = max(time_dd, curr)

    # Trade stats
    if trades is None or trades.empty:
        n_trades = 0
        win_rate = 0.0
        rr = 0.0
        avg_hold_bars = 0
    else:
        n_trades = len(trades)
        wins = trades.loc[trades["pnl"] > 0, "pnl"].values
        losses = trades.loc[trades["pnl"] < 0, "pnl"].abs().values
        win_rate = (len(wins) / n_trades) * 100.0 if n_trades else 0.0
        avg_win = np.mean(wins) if len(wins) else 0.0
        avg_loss = np.mean(losses) if len(losses) else np.nan
        rr = (avg_win / avg_loss) if avg_loss and not np.isnan(avg_loss) and avg_loss > 0 else 0.0
        avg_hold_bars = trades.apply(lambda r: (r["exit_time"] - r["entry_time"]).total_seconds() / 60, axis=1).mean()

    # Sharpe ratio (approx, using bar returns)
    returns = equity.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if not returns.empty else 0.0

    return {
        "n_trades": int(n_trades),
        "win_rate": float(round(win_rate, 2)),
        "roi_pct": float(round(roi, 2)),
        "max_dd_pct": float(round(max_dd, 2)),
        "time_dd_bars": int(time_dd),
        "rr": float(round(rr, 2)),
        "avg_hold_minutes": float(round(avg_hold_bars, 2)),
        "sharpe_ratio": float(round(sharpe, 2)),
    }
