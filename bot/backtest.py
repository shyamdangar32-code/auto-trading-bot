# bot/backtest.py
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .strategy import (
    prepare_signals, initial_stop_target, trail_stop,
    LONG, SHORT, FLAT
)
from .metrics import compute_metrics  # ⬅️ NEW: centralize metrics here


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
      - Re-entry (cooldown supported)
      - ATR stop/target
      - Trailing stop (ATR-based via strategy.trail_stop)
    Returns: (summary_dict, trades_df, equity_series)
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
        atr_val = float(row["atr"]) if not np.isnan(row.get("atr", np.nan)) else 0.0

        # trailing while in position
        if position != FLAT:
            stop = trail_stop(position, px, atr_val, stop, entry_px, cfg)

        # -------- exits --------
        did_exit = False
        if position == LONG:
            # stop
            if row["Low"] <= stop:
                trades[-1].exit_time = ts
                trades[-1].exit = stop
                trades[-1].reason = "STOP"
                trades[-1].pnl = (stop - entry_px) * qty
                equity += trades[-1].pnl
                position = FLAT
                did_exit = True
            # target
            elif row["High"] >= target:
                trades[-1].exit_time = ts
                trades[-1].exit = target
                trades[-1].reason = "TARGET"
                trades[-1].pnl = (target - entry_px) * qty
                equity += trades[-1].pnl
                position = FLAT
                did_exit = True

        elif position == SHORT:
            # stop
            if row["High"] >= stop:
                trades[-1].exit_time = ts
                trades[-1].exit = stop
                trades[-1].reason = "STOP"
                trades[-1].pnl = (entry_px - stop) * qty
                equity += trades[-1].pnl
                position = FLAT
                did_exit = True
            # target
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
            # keep re_count; it’s handled on new-day reset

        # -------- entries & re-entries --------
        if position == FLAT:
            ok_after_cooldown = (i - last_exit_idx) >= cooldown

            if ok_after_cooldown:
                if bool(row.get("long_entry", False)) and (re_count < re_max or re_max == 0):
                    position = LONG
                    entry_px = px
                    stop, target = initial_stop_target(LONG, entry_px, atr_val, cfg)
                    trades.append(Trade(LONG, ts, entry_px))
                    # if this is a re-entry (i.e., there was a prior exit in the day)
                    re_count += 1 if last_exit_idx > -10**8 else 0

                elif bool(row.get("short_entry", False)) and (re_count < re_max or re_max == 0):
                    position = SHORT
                    entry_px = px
                    stop, target = initial_stop_target(SHORT, entry_px, atr_val, cfg)
                    trades.append(Trade(SHORT, ts, entry_px))
                    re_count += 1 if last_exit_idx > -10**8 else 0

        # -------- new day => reset re-entry counter --------
        if i > 0:
            prev_day = pd.Timestamp(idx_list[i-1]).date()
            this_day = pd.Timestamp(ts).date()
            if this_day != prev_day:
                re_count = 0

        # mark-to-market (simple)
        eq_curve.append(equity)

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    if not trades_df.empty:
        trades_df["side"] = trades_df["side"].map({1: "LONG", -1: "SHORT"})

    equity_ser = pd.Series(eq_curve, index=d.index, name="equity")

    # 🔢 NEW: richer metrics from bot/metrics.py
    summary = compute_metrics(trades_df, equity_ser, capital)
    return summary, trades_df, equity_ser


def save_reports(out_dir: str, summary: dict, trades: pd.DataFrame, equity: pd.Series):
    """
    Persist backtest artifacts:
      - trades.csv
      - equity.csv
      - metrics.json
    """
    out_dir = out_dir.rstrip("/")

    if trades is not None and not trades.empty:
        trades.to_csv(f"{out_dir}/trades.csv", index=False)

    if equity is not None and not equity.empty:
        equity.to_csv(f"{out_dir}/equity.csv", header=True)

    with open(f"{out_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary or {}, f, indent=2)
