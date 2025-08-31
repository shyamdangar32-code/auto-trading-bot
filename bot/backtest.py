# bot/backtest.py
from __future__ import annotations
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .strategy import (
    prepare_signals, initial_stop_target, trail_stop,
    LONG, SHORT, FLAT
)
from .metrics import compute_metrics
from .evaluation import plot_equity_and_drawdown, write_quick_report


@dataclass
class Trade:
    side: int
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    reason: str = ""
    pnl: float = 0.0


def _day_key(ts: pd.Timestamp) -> pd.Timestamp.date:
    return (ts.tz_localize(None) if ts.tzinfo else ts).date()


def run_backtest(prices: pd.DataFrame, cfg: dict, use_block: str = "backtest"):
    """
    Walk-forward backtest:
      - ORB + VWAP entries
      - ATR stop/target
      - Trailing stop
      - Re-entry with cooldown
      - Day guardrails: max trades/day, max daily loss, optional daily target stop
    Returns: (summary_dict, trades_df, equity_series)
    """
    block = (cfg.get(use_block) or cfg.get("intraday_options") or {})
    entry_cfg     = block.get("entry", {})
    exits_cfg     = block.get("exits", {})
    reentry_cfg   = block.get("reentry", {})
    guards_cfg    = block.get("guardrails", {})

    d = prepare_signals(prices, cfg, use_block=use_block).copy()

    qty        = int(cfg.get("order_qty", 1))
    capital    = float(cfg.get("capital_rs", 100000.0))
    re_max     = int(reentry_cfg.get("max_per_day", 0))
    cooldown   = int(reentry_cfg.get("cooldown_bars", 0))

    max_trades_day   = int(guards_cfg.get("max_trades_per_day", 9999))
    max_daily_loss   = float(guards_cfg.get("max_daily_loss_rs", 1e18))
    stop_after_target= float(guards_cfg.get("stop_after_target_rs", 1e18))

    position = FLAT
    entry_px = stop = target = np.nan
    re_count = 0
    last_exit_idx = -10**9
    trades: list[Trade] = []

    equity_val = capital
    eq_curve: list[float] = []

    # day state
    day_trade_count = 0
    day_pnl_accum   = 0.0
    block_new_entries_today = False
    prev_day = None

    idx_list = list(d.index)
    for i, ts in enumerate(idx_list):
        row = d.loc[ts]
        px  = float(row["Close"])
        atr_val = float(row.get("atr", np.nan)) if not np.isnan(row.get("atr", np.nan)) else 0.0

        # day rollover
        this_day = _day_key(ts)
        if prev_day is None:
            prev_day = this_day
        if this_day != prev_day:
            # reset per-day counters
            day_trade_count = 0
            day_pnl_accum   = 0.0
            re_count        = 0
            block_new_entries_today = False
            prev_day = this_day

        # trailing while in position
        if position != FLAT:
            stop = trail_stop(position, px, atr_val, stop, entry_px, exits_cfg)

        # ----- exits -----
        did_exit = False
        if position == LONG:
            if row["Low"] <= stop:
                tr = trades[-1]; tr.exit_time = ts; tr.exit = stop; tr.reason = "STOP"
                tr.pnl = (stop - entry_px) * qty; equity_val += tr.pnl
                day_pnl_accum += tr.pnl
                position = FLAT; did_exit = True
            elif row["High"] >= target:
                tr = trades[-1]; tr.exit_time = ts; tr.exit = target; tr.reason = "TARGET"
                tr.pnl = (target - entry_px) * qty; equity_val += tr.pnl
                day_pnl_accum += tr.pnl
                position = FLAT; did_exit = True

        elif position == SHORT:
            if row["High"] >= stop:
                tr = trades[-1]; tr.exit_time = ts; tr.exit = stop; tr.reason = "STOP"
                tr.pnl = (entry_px - stop) * qty; equity_val += tr.pnl
                day_pnl_accum += tr.pnl
                position = FLAT; did_exit = True
            elif row["Low"] <= target:
                tr = trades[-1]; tr.exit_time = ts; tr.exit = target; tr.reason = "TARGET"
                tr.pnl = (entry_px - target) * qty; equity_val += tr.pnl
                day_pnl_accum += tr.pnl
                position = FLAT; did_exit = True

        if did_exit:
            last_exit_idx = i
            # guardrails after exit
            if day_pnl_accum <= -abs(max_daily_loss):
                block_new_entries_today = True
            if day_pnl_accum >= abs(stop_after_target):
                block_new_entries_today = True

        # ----- entries & re-entries -----
        if position == FLAT and not block_new_entries_today:
            ok_after_cooldown = (i - last_exit_idx) >= cooldown
            within_trade_cap  = (day_trade_count < max_trades_day)

            if ok_after_cooldown and within_trade_cap:
                if bool(row.get("long_entry", False)) and (re_count < re_max or re_max == 0):
                    position = LONG
                    entry_px = px
                    stop, target = initial_stop_target(LONG, entry_px, atr_val, exits_cfg)
                    trades.append(Trade(LONG, ts, entry_px))
                    re_count += 1 if last_exit_idx > -10**8 else 0
                    day_trade_count += 1

                elif bool(row.get("short_entry", False)) and (re_count < re_max or re_max == 0):
                    position = SHORT
                    entry_px = px
                    stop, target = initial_stop_target(SHORT, entry_px, atr_val, exits_cfg)
                    trades.append(Trade(SHORT, ts, entry_px))
                    re_count += 1 if last_exit_idx > -10**8 else 0
                    day_trade_count += 1

            # if cap hit, freeze further entries for the day
            if day_trade_count >= max_trades_day:
                block_new_entries_today = True

        eq_curve.append(equity_val)

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    if not trades_df.empty:
        trades_df["side"] = trades_df["side"].map({1: "LONG", -1: "SHORT"})

    equity_ser = pd.Series(eq_curve, index=d.index, name="equity")

    # metrics
    summary = compute_metrics(trades_df, equity_ser, capital)
    return summary, trades_df, equity_ser


def save_reports(out_dir: str, summary: dict, trades: pd.DataFrame, equity: pd.Series):
    """
    Writes standard artifacts + quick report & charts.
    """
    out_dir = out_dir.rstrip("/")

    if trades is not None and not trades.empty:
        trades.to_csv(f"{out_dir}/trades.csv", index=False)

    if equity is not None and not equity.empty:
        equity.to_csv(f"{out_dir}/equity.csv", header=True)

    with open(f"{out_dir}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary or {}, f, indent=2)

    if equity is not None and not equity.empty:
        plot_equity_and_drawdown(equity, out_dir)
    write_quick_report(summary or {}, trades, out_dir)
