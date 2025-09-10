# bot/backtest.py  (MIN-HOLD + COOLDOWN + EOD SQUARE-OFF SAME-DAY, TZ-SAFE)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class Trade:
    side: int
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp | None = None
    exit: float | None = None
    qty: float = 0.0
    pnl: float = 0.0
    reason: str = ""

LONG, SHORT, FLAT = +1, -1, 0

def _first_hit_long(row: pd.Series, stop: float, target: float):
    if row["low"] <= stop:   return stop, "SL"
    if row["high"] >= target:return target, "TP"
    return None, None

def _first_hit_short(row: pd.Series, stop: float, target: float):
    if row["high"] >= stop:  return stop, "SL"
    if row["low"]  <= target:return target, "TP"
    return None, None

def _to_market_tz(ts: pd.Timestamp, market_tz: str) -> pd.Timestamp:
    """If ts tz-aware -> convert; if naive -> return as-is (assume already market tz)."""
    try:
        if ts.tz is None:
            return ts
        return ts.tz_convert(market_tz)
    except Exception:
        return ts

def run_backtest(df_in: pd.DataFrame, cfg: Dict, use_block: str = "backtest_loose") -> Tuple[Dict, pd.DataFrame, pd.Series]:
    df = df_in.copy()
    # normalize expected columns
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            cap = c.capitalize()
            df[c] = df[cap] if cap in df.columns else np.nan
    df.dropna(subset=["open","high","low","close","signal"], inplace=True)

    # merge plan
    plan: Dict = {}
    if "backtest" in cfg: plan.update(cfg["backtest"] or {})
    if use_block in cfg:  plan.update(cfg[use_block] or {})

    capital = float(cfg.get("capital_rs", cfg.get("capital", plan.get("capital_rs", 100000))))
    fallback_qty = int(cfg.get("order_qty", plan.get("order_qty", 1)))
    min_hold = int(plan.get("min_hold_bars", 0))
    cooldown_bars = int(plan.get("cooldown_bars", 0))
    allow_trail = bool(plan.get("trail_after_hold", True))

    # EOD settings
    sess_end_str = str(plan.get("session_end", "15:20"))
    try:
        sess_end_time = pd.to_datetime(sess_end_str).time()
    except Exception:
        sess_end_time = pd.to_datetime("15:20").time()
    market_tz = str(plan.get("market_tz", plan.get("tz", "Asia/Kolkata")))

    position = FLAT
    entry_px = np.nan
    qty = 0
    stop = np.nan
    target = np.nan
    cool = 0
    hold = 0

    eq = pd.Series(index=df.index, dtype=float); eq.iloc[0] = capital
    trades: list[Trade] = []
    last = capital

    for i, (ts, row) in enumerate(df.iterrows()):
        if i == 0:
            continue

        ts_mkt = _to_market_tz(ts, market_tz)

        if position == FLAT:
            if cool > 0:
                cool -= 1
            else:
                sig = int(row["signal"])
                if sig != 0:
                    position = LONG if sig > 0 else SHORT
                    entry_px = float(row["open"])
                    qty = int(row.get("pos_size", fallback_qty) or fallback_qty)
                    stop = float(row.get("sl_px", np.nan))
                    target = float(row.get("tp_px", np.nan))
                    hold = 0
                    trades.append(Trade(side=position, entry_time=ts, entry=entry_px, qty=qty))
        else:
            hold += 1

            # --- HARD guard: if date rolled (overnight), exit at this bar OPEN (carry cleanup) ---
            entry_mkt = _to_market_tz(trades[-1].entry_time, market_tz)
            if ts_mkt.date() > entry_mkt.date():
                exit_px, reason = float(row["open"]), "EOD_CARRY"
                pnl = (exit_px - entry_px) * qty if position == LONG else (entry_px - exit_px) * qty
                last += pnl
                trades[-1].exit_time = ts; trades[-1].exit = exit_px; trades[-1].pnl = pnl; trades[-1].reason = reason
                position = FLAT; cool = cooldown_bars; hold = 0
                entry_px = np.nan; qty = 0; stop = np.nan; target = np.nan
                eq.iloc[i] = last
                continue

            # --- SAME-DAY EOD square-off: at/after session_end -> exit at THIS bar OPEN ---
            if ts_mkt.time() >= sess_end_time:
                exit_px, reason = float(row["open"]), "EOD"
                pnl = (exit_px - entry_px) * qty if position == LONG else (entry_px - exit_px) * qty
                last += pnl
                trades[-1].exit_time = ts; trades[-1].exit = exit_px; trades[-1].pnl = pnl; trades[-1].reason = reason
                position = FLAT; cool = cooldown_bars; hold = 0
                entry_px = np.nan; qty = 0; stop = np.nan; target = np.nan
                eq.iloc[i] = last
                continue

            # --- normal exit logic after min-hold ---
            exit_px = None; reason = None
            if hold >= min_hold:
                if position == LONG:
                    exit_px, reason = _first_hit_long(row, stop, target)
                else:
                    exit_px, reason = _first_hit_short(row, stop, target)

            # --- trailing after min-hold ---
            if (exit_px is None) and allow_trail and (hold >= min_hold):
                atr_val = row.get("atr", np.nan)
                if np.isfinite(atr_val):
                    if position == LONG:
                        stop = max(stop, row["close"] - 1.0 * atr_val)
                    else:
                        stop = min(stop, row["close"] + 1.0 * atr_val)

            if exit_px is not None:
                pnl = (exit_px - entry_px) * qty if position == LONG else (entry_px - exit_px) * qty
                last += pnl
                trades[-1].exit_time = ts; trades[-1].exit = exit_px; trades[-1].pnl = pnl; trades[-1].reason = reason
                position = FLAT; cool = cooldown_bars; hold = 0
                entry_px = np.nan; qty = 0; stop = np.nan; target = np.nan

        eq.iloc[i] = last

    tdf = pd.DataFrame([t.__dict__ for t in trades]); tdf.index.name = "trade_id"

    gross_profit = tdf.loc[tdf["pnl"] > 0, "pnl"].sum() if not tdf.empty else 0.0
    gross_loss = -tdf.loc[tdf["pnl"] < 0, "pnl"].sum() if not tdf.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    wins = int((tdf["pnl"] > 0).sum()) if not tdf.empty else 0
    total = int(len(tdf))
    win_rate = (wins / total * 100.0) if total else 0.0
    cum_roi = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0 if len(eq) > 1 else 0.0

    roll_max = eq.cummax(); dd = (eq - roll_max)
    max_dd_abs = dd.min() if len(dd) else 0.0
    max_dd_perc = (max_dd_abs / roll_max.loc[dd.idxmin()]) * 100.0 if len(dd) and roll_max.loc[dd.idxmin()] != 0 else 0.0
    rets = eq.pct_change().fillna(0.0)
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252*390) if rets.std() > 0 else 0.0

    summary = dict(
        trades=total,
        win_rate=round(win_rate, 2),
        ROI=round(cum_roi, 2),
        profit_factor=round(float(profit_factor), 2) if np.isfinite(profit_factor) else float("inf"),
        R_R=0.0,
        max_dd_perc=round(max_dd_perc, 2),
        time_dd_bars=int((eq.index.size - (eq.cummax()==eq).sum())),
        sharpe=round(float(sharpe), 2),
    )
    return summary, tdf, eq

def _plot_equity(eq: pd.Series, outdir: Path):
    fig, ax = plt.subplots(figsize=(10,4))
    eq.plot(ax=ax); ax.set_title("Equity Curve"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); (outdir / "equity_curve.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "equity_curve.png", dpi=120); plt.close(fig)

    roll_max = eq.cummax(); dd = eq - roll_max
    fig2, ax2 = plt.subplots(figsize=(10,2.5))
    dd.plot(ax=ax2); ax2.set_title("Drawdown"); ax2.grid(True, alpha=0.3)
    fig2.tight_layout(); fig2.savefig(outdir / "drawdown.png", dpi=120); plt.close(fig2)

def save_reports(outdir: str | Path, summary: Dict, trades_df: pd.DataFrame, equity_ser: pd.Series) -> None:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out / "trades.csv", index=True)
    equity_ser.to_csv(out / "equity.csv", index=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_equity(equity_ser, out)

    md_lines = [
        "# Backtest Summary\n",
        f"- **Trades**: {summary['trades']}",
        f"- **Win-rate**: {summary['win_rate']}%",
        f"- **ROI**: {summary['ROI']}%",
        f"- **Profit Factor**: {summary['profit_factor']}",
        f"- **Max DD**: {summary['max_dd_perc']}%",
        f"- **Time DD (bars)**: {summary['time_dd_bars']}",
        f"- **Sharpe**: {summary['sharpe']}",
        "\n![Equity](equity_curve.png)\n",
        "![Drawdown](drawdown.png)\n",
    ]
    (out / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
