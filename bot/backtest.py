# bot/backtest.py
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .strategy import (
    prepare_signals, initial_stop_target, trail_stop,
    LONG, SHORT, FLAT
)

@dataclass
class Trade:
    side: int
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp = None
    exit: float = None
    reason: str = ""
    pnl: float = 0.0

def run_backtest(prices: pd.DataFrame, cfg: dict):
    """
    Walk-forward backtest with:
      - Re-entry
      - ATR stop/target
      - Trailing stop
    Returns (summary_dict, trades_df, equity_series)
    """
    d = prepare_signals(prices, cfg).copy()

    qty        = int(cfg.get("order_qty", 1))
    capital    = float(cfg.get("capital_rs", 100000.0))
    re_max     = int(cfg.get("reentry_max", 0))
    cooldown   = int(cfg.get("reentry_cooldown", 0))

    # containers
    position = FLAT
    entry_px = stop = target = np.nan
    re_count = 0
    last_exit_idx = -10**9
    trades: list[Trade] = []

    equity = capital
    eq_curve = []

    idx_list = list(d.index)
    for i, ts in enumerate(idx_list):
        row = d.loc[ts]
        px  = float(row["Close"])
        atr = float(row["atr"]) if not np.isnan(row.get("atr", np.nan)) else 0.0

        # update trailing if in position
        if position != FLAT:
            stop = trail_stop(position, px, atr, stop, entry_px, cfg)

        # exits
        did_exit = False
        if position == LONG:
            if row["Low"] <= stop:
                trades[-1].exit_time = ts
                trades[-1].exit = stop
                trades[-1].reason = "STOP"
                trades[-1].pnl = (stop - entry_px) * qty
                equity += trades[-1].pnl
                position = FLAT
                did_exit = True
            elif row["High"] >= target:
                trades[-1].exit_time = ts
                trades[-1].exit = target
                trades[-1].reason = "TARGET"
                trades[-1].pnl = (target - entry_px) * qty
                equity += trades[-1].pnl
                position = FLAT
                did_exit = True

        elif position == SHORT:
            if row["High"] >= stop:
                trades[-1].exit_time = ts
                trades[-1].exit = stop
                trades[-1].reason = "STOP"
                trades[-1].pnl = (entry_px - stop) * qty
                equity += trades[-1].pnl
                position = FLAT
                did_exit = True
            elif row["Low"] <= target:
                trades[-1].exit_time = ts
                trades[-1].exit = target
                trades[-1].reason = "TARGET"
                trades[-1].pnl = (entry_px - target) * qty
                equity += trades[-1].pnl
                position = FLAT
                did_exit = True

        if did_exit:
            last_exit_idx = i
            # allow re-entries later; counter preserved

        # entries & re-entries
        if position == FLAT:
            same_day = True
            if i - last_exit_idx < cooldown:
                same_day = False  # still in cooldown window

            if same_day:
                if d.loc[ts, "long_entry"] and (re_count < re_max or re_max == 0):
                    position = LONG
                    entry_px = px
                    stop, target = initial_stop_target(LONG, entry_px, atr, cfg)
                    trades.append(Trade(LONG, ts, entry_px))
                    re_count += 1 if last_exit_idx > -10**9 else 0

                elif d.loc[ts, "short_entry"] and (re_count < re_max or re_max == 0):
                    position = SHORT
                    entry_px = px
                    stop, target = initial_stop_target(SHORT, entry_px, atr, cfg)
                    trades.append(Trade(SHORT, ts, entry_px))
                    re_count += 1 if last_exit_idx > -10**9 else 0

        # new day -> reset re-entry counter
        if i > 0:
            prev_day = pd.Timestamp(idx_list[i-1]).date()
            this_day = pd.Timestamp(ts).date()
            if this_day != prev_day:
                re_count = 0

        eq_curve.append(equity if position == FLAT else equity)  # equity shown mark-to-market

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    if not trades_df.empty:
        trades_df["side"] = trades_df["side"].map({1: "LONG", -1: "SHORT"})
    equity_ser = pd.Series(eq_curve, index=d.index, name="equity")

    summary = evaluate(equity_ser, trades_df, capital)
    return summary, trades_df, equity_ser

def evaluate(equity: pd.Series, trades: pd.DataFrame, starting_capital: float):
    """Compute ROI, Drawdown, R:R, Win-rate, Equity curve stats, Time drawdown."""
    if equity.empty:
        return {"n_trades": 0, "win_rate": 0.0, "roi_pct": 0.0,
                "max_dd_pct": 0.0, "time_dd_bars": 0, "rr": 0.0}

    # ROI
    roi = (equity.iloc[-1] - starting_capital) / starting_capital * 100.0

    # Drawdown (magnitude)
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = dd.min() * 100.0

    # Time drawdown (longest stretch below prior peak)
    time_dd = 0
    curr = 0
    for below in (equity < roll_max):
        curr = curr + 1 if below else 0
        time_dd = max(time_dd, curr)

    # Trade stats
    if trades.empty:
        win_rate = rr = 0.0
    else:
        wins = trades.loc[trades["pnl"] > 0, "pnl"].values
        losses = trades.loc[trades["pnl"] < 0, "pnl"].abs().values
        win_rate = (len(wins) / len(trades)) * 100.0 if len(trades) else 0.0
        avg_win = np.mean(wins) if len(wins) else 0.0
        avg_loss = np.mean(losses) if len(losses) else np.nan
        rr = (avg_win / avg_loss) if avg_loss and not np.isnan(avg_loss) and avg_loss > 0 else 0.0

    return {
        "n_trades": int(len(trades)),
        "win_rate": float(round(win_rate, 2)),
        "roi_pct": float(round(roi, 2)),
        "max_dd_pct": float(round(max_dd, 2)),
        "time_dd_bars": int(time_dd),
        "rr": float(round(rr, 2)),
    }

def save_reports(out_dir: str, summary: dict, trades: pd.DataFrame, equity: pd.Series):
    out_dir = out_dir.rstrip("/")

    if trades is not None and not trades.empty:
        trades.to_csv(f"{out_dir}/trades.csv", index=False)
    equity.to_csv(f"{out_dir}/equity.csv", header=True)

    with open(f"{out_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
