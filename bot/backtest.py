# bot/backtest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Trade:
    side: int            # +1 long, -1 short
    entry_idx: int
    exit_idx: int
    entry_px: float
    exit_px: float
    qty: int
    pnl: float
    reason: str          # 'tp'/'sl'/'rev'/'eod'


def _first_hit_long(row) -> str:
    """Decide which was hit first for a long position (pessimistic: SL before TP if both)."""
    low, high, sl, tp = row["low"], row["high"], row["sl_px"], row["tp_px"]
    hit_sl = np.isfinite(sl) and (low <= sl)
    hit_tp = np.isfinite(tp) and (high >= tp)
    if hit_sl and hit_tp:
        return "sl"  # pessimistic
    if hit_sl:
        return "sl"
    if hit_tp:
        return "tp"
    return ""


def _first_hit_short(row) -> str:
    low, high, sl, tp = row["low"], row["high"], row["sl_px"], row["tp_px"]
    hit_sl = np.isfinite(sl) and (high >= sl)
    hit_tp = np.isfinite(tp) and (low <= tp)
    if hit_sl and hit_tp:
        return "sl"
    if hit_sl:
        return "sl"
    if hit_tp:
        return "tp"
    return ""


def run_backtest(
    df_in: pd.DataFrame,
    cfg: Dict,
    use_block: str = "backtest_loose",
) -> Tuple[Dict, pd.DataFrame, pd.Series]:
    """
    Minimal, deterministic bar-by-bar backtest that respects:
      - signal (+1/-1/0)
      - sl_px / tp_px
      - pos_size (optional; falls back to cfg['order_qty'])
    Expected columns: index as datetime; open/high/low/close lowercased.
    """
    df = df_in.copy()
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            df[c] = df[c.capitalize()] if c.capitalize() in df.columns else np.nan
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    order_qty = int(cfg.get("order_qty", 1))
    eod_squareoff = True  # intraday assumption

    in_pos = 0           # +1 long, -1 short, 0 flat
    entry_px = np.nan
    qty = 0
    cooldown_left = 0
    trades: list[Trade] = []
    equity = [float(cfg.get("capital_rs", 100000.0))]
    capital = equity[0]

    # iterate from second bar to allow next-bar execution
    for i in range(1, len(df)):
        prev, row = df.iloc[i - 1], df.iloc[i]
        signal = int(prev.get("signal", 0))
        sl_px = float(prev.get("sl_px", np.nan))
        tp_px = float(prev.get("tp_px", np.nan))
        bar_open = float(row["open"])

        # manage open position (SL/TP on *current* bar)
        exit_reason = ""
        if in_pos == 1:
            exit_reason = _first_hit_long(row)
        elif in_pos == -1:
            exit_reason = _first_hit_short(row)

        if exit_reason:
            exit_px = sl_px if exit_reason == "sl" else tp_px
            exit_px = float(exit_px) if np.isfinite(exit_px) else float(row["close"])
            pnl = (exit_px - entry_px) * qty * in_pos
            trades.append(Trade(in_pos, i - 1, i, entry_px, exit_px, qty, pnl, exit_reason))
            capital += pnl
            equity.append(capital)
            in_pos, entry_px, qty = 0, np.nan, 0
            cooldown_left = int(prev.get("cooldown_bars", 0))
        else:
            equity.append(capital)

        # optional end-of-day square-off (when date changes)
        if eod_squareoff and in_pos != 0:
            if df.index[i].date() != df.index[i - 1].date():
                exit_px = float(prev["close"])
                pnl = (exit_px - entry_px) * qty * in_pos
                trades.append(Trade(in_pos, i - 1, i - 1, entry_px, exit_px, qty, pnl, "eod"))
                capital += pnl
                equity[-1] = capital
                in_pos, entry_px, qty = 0, np.nan, 0
                cooldown_left = int(prev.get("cooldown_bars", 0))

        # entry (next-bar open) only if flat and not in cooldown
        if in_pos == 0 and cooldown_left <= 0:
            if signal != 0:
                sized_qty = int(prev.get("pos_size", order_qty))
                sized_qty = max(1, sized_qty)
                entry_px = bar_open
                in_pos = int(np.sign(signal))
                qty = sized_qty
        else:
            cooldown_left = max(0, cooldown_left - 1)

    # Trades DataFrame
    tdf = pd.DataFrame([t.__dict__ for t in trades])
    tdf.index.name = "trade_id"

    # equity series aligned to df index length
    eq = pd.Series(equity[: len(df)], index=df.index[: len(equity)])

    # ---- metrics ----
    gross_profit = tdf.loc[tdf["pnl"] > 0, "pnl"].sum() if not tdf.empty else 0.0
    gross_loss = -tdf.loc[tdf["pnl"] < 0, "pnl"].sum() if not tdf.empty else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)

    wins = (tdf["pnl"] > 0).sum() if not tdf.empty else 0
    total = len(tdf)
    win_rate = (wins / total * 100.0) if total else 0.0

    returns = pd.Series(eq).pct_change().fillna(0.0)
    cum_roi = (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0 if len(eq) > 1 else 0.0
    # drawdown
    roll_max = eq.cummax()
    dd = (eq - roll_max)
    max_dd_abs = dd.min() if len(dd) else 0.0
    max_dd_perc = (max_dd_abs / roll_max.loc[dd.idxmin()]) * 100.0 if len(dd) and roll_max.loc[dd.idxmin()] != 0 else 0.0
    # time in drawdown (bars)
    time_dd = 0
    cur = 0
    for v, m in zip(eq, roll_max):
        if v < m:
            cur += 1
            time_dd = max(time_dd, cur)
        else:
            cur = 0

    # avg R:R (approx)
    avg_win = tdf.loc[tdf["pnl"] > 0, "pnl"].mean() if wins else np.nan
    avg_loss = -tdf.loc[tdf["pnl"] < 0, "pnl"].mean() if (total - wins) else np.nan
    rr = (avg_win / avg_loss) if (np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss > 0) else 0.0

    sharpe = 0.0
    if returns.std(ddof=1) not in (0, np.nan):
        sharpe = float(np.sqrt(252) * returns.mean() / (returns.std(ddof=1) + 1e-12))

    summary = {
        "trades": int(total),
        "win_rate": round(win_rate, 2),
        "ROI": round(cum_roi, 2),
        "profit_factor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else float("inf"),
        "R:R": round(rr, 2) if rr else 0.0,
        "max_dd_perc": round(float(max_dd_perc), 2),
        "time_dd_bars": int(time_dd),
        "sharpe": round(sharpe, 2) if np.isfinite(sharpe) else 0.0,
    }

    return summary, tdf, eq


# ---------------- reporting helpers ---------------- #

def _plot_equity(eq: pd.Series, out: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(eq.index, eq.values)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (₹)")
    plt.tight_layout()
    plt.savefig(out / "equity_curve.png", dpi=140)
    plt.close()

    # drawdown
    roll_max = eq.cummax()
    dd = eq - roll_max
    plt.figure(figsize=(7, 3.5))
    plt.plot(eq.index, dd.values)
    plt.title("Drawdown (₹)")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out / "drawdown.png", dpi=140)
    plt.close()


def save_reports(outdir: Path | str, summary: Dict, trades_df: pd.DataFrame, equity_ser: pd.Series) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # CSVs
    if not trades_df.empty:
        trades_df.to_csv(out / "trades.csv", index=True)
    equity_ser.to_csv(out / "equity.csv", header=["equity"])

    # Plots
    _plot_equity(equity_ser, out)

    # metrics.json + pretty markdown
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Backtest Summary\n",
        f"- **Trades**: {summary['trades']}",
        f"- **Win-rate**: {summary['win_rate']}%",
        f"- **ROI**: {summary['ROI']}%",
        f"- **Profit Factor**: {summary['profit_factor']}",
        f"- **R:R**: {summary['R:R']}",
        f"- **Max DD**: {summary['max_dd_perc']}%",
        f"- **Time DD (bars)**: {summary['time_dd_bars']}",
        f"- **Sharpe**: {summary['sharpe']}",
        "\n![Equity](equity_curve.png)\n",
        "![Drawdown](drawdown.png)\n",
    ]
    (out / "REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")
