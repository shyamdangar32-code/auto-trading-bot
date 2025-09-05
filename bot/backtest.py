# bot/backtest.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import time as dtime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))

def _within_session(ts: pd.Timestamp, sess_start: str, sess_end: str) -> bool:
    """
    Pandas 2.x પર pd.Timestamp(hour=9,minute=15) થી error આવતો હતો,
    એટલા માટે datetime.time થી compare કરીએ છીએ.
    """
    tt = ts.tz_convert(None).time() if ts.tzinfo else ts.time()
    s = _parse_hhmm(sess_start)
    e = _parse_hhmm(sess_end)
    return (tt >= s) and (tt <= e)


@dataclass
class Position:
    side: str = ""         # "long" only for now
    entry_px: float = 0.0
    sl_px: float = 0.0
    tp_px: float = 0.0
    qty: int = 0
    entry_ts: pd.Timestamp | None = None


# -------------------------
# Engine
# -------------------------

def run_backtest(df: pd.DataFrame, cfg: dict, use_block: str = "backtest_loose"):
    """
    Very simple long-only engine using SL/TP in ATR multiples. Exits at SL/TP or end-of-day.
    Returns: (summary: dict, trades_df: DataFrame, equity_ser: Series)
    df must already include columns from prepare_signals(): enter_long, atr, etc.
    """
    df = df.copy()

    # parameters
    bcfg = (cfg.get("backtest") or {})
    sess_s = (bcfg.get("filters", {}) or {}).get("session_start", bcfg.get("session_start", "09:20"))
    sess_e = (bcfg.get("filters", {}) or {}).get("session_end",   bcfg.get("session_end",   "15:20"))
    stop_mult = float(((bcfg.get("exits") or {}).get("stop_atr_mult", cfg.get("stop_atr_mult", 1.0))))
    take_mult = float(((bcfg.get("exits") or {}).get("take_atr_mult", cfg.get("take_atr_mult", 1.3))))

    capital = float(cfg.get("capital_rs", 100000.0))
    qty     = int(cfg.get("order_qty", 1))

    # local state
    pos = None
    cash = capital
    eq_curve = []
    trades = []

    for ts, row in df.iterrows():
        # session filter
        if not _within_session(ts, sess_s, sess_e):
            # square-off at session end
            if pos is not None:
                pnl = (row["Close"] - pos.entry_px) * pos.qty
                cash += pnl
                trades.append(dict(
                    entry_ts=pos.entry_ts, exit_ts=ts, side="long",
                    entry=pos.entry_px, exit=row["Close"], qty=pos.qty, pnl=pnl
                ))
                pos = None
            eq_curve.append((ts, cash))
            continue

        # entry
        if pos is None and bool(row.get("enter_long", False)):
            entry = float(row["Close"])
            atr   = float(row.get("atr", 0.0))
            sl    = entry - stop_mult * atr
            tp    = entry + take_mult * atr
            pos = Position(side="long", entry_px=entry, sl_px=sl, tp_px=tp, qty=qty, entry_ts=ts)

        # manage open
        if pos is not None:
            low = float(row.get("Low", row["Close"]))
            high = float(row.get("High", row["Close"]))
            exit_px = None
            reason = None
            if low <= pos.sl_px:
                exit_px = pos.sl_px; reason = "SL"
            elif high >= pos.tp_px:
                exit_px = pos.tp_px; reason = "TP"

            if exit_px is not None:
                pnl = (exit_px - pos.entry_px) * pos.qty
                cash += pnl
                trades.append(dict(
                    entry_ts=pos.entry_ts, exit_ts=ts, side="long",
                    entry=pos.entry_px, exit=exit_px, qty=pos.qty, pnl=pnl, reason=reason
                ))
                pos = None

        eq_curve.append((ts, cash))

    # finalize: square-off if still open at last bar
    if pos is not None:
        last_ts = df.index[-1]
        last_px = float(df.iloc[-1]["Close"])
        pnl = (last_px - pos.entry_px) * pos.qty
        cash += pnl
        trades.append(dict(
            entry_ts=pos.entry_ts, exit_ts=last_ts, side="long",
            entry=pos.entry_px, exit=last_px, qty=pos.qty, pnl=pnl, reason="EOD"
        ))
        pos = None
        eq_curve[-1] = (last_ts, cash)

    equity_ser = pd.Series({ts: val for ts, val in eq_curve}).sort_index()
    base = capital
    ret = (equity_ser.iloc[-1] - base) / base if len(equity_ser) else 0.0

    trades_df = pd.DataFrame(trades)
    win = float((trades_df["pnl"] > 0).mean()*100) if not trades_df.empty else 0.0
    rr  = (trades_df.loc[trades_df["pnl"]>0,"pnl"].mean() / abs(trades_df.loc[trades_df["pnl"]<0,"pnl"].mean())) if ((trades_df["pnl"]>0).any() and (trades_df["pnl"]<0).any()) else 0.0
    pf  = (trades_df.loc[trades_df["pnl"]>0,"pnl"].sum() / abs(trades_df.loc[trades_df["pnl"]<0,"pnl"].sum())) if ((trades_df["pnl"]>0).any() and (trades_df["pnl"]<0).any()) else 0.0

    # drawdown
    roll_max = equity_ser.cummax()
    dd = equity_ser - roll_max
    max_dd = float((dd.min() / base) * 100) if len(dd) else 0.0

    summary = dict(
        n_trades=int(len(trades_df)),
        win_rate=round(win, 2),
        roi_pct=round(ret*100, 2),
        profit_factor=round(pf, 2),
        rr=round(rr, 2),
        sharpe_ratio=round(0.0, 2),        # placeholder
        max_dd_pct=round(max_dd, 2),
        time_dd_bars=int((dd == dd.min()).sum()) if len(dd) else 0,
        n_bars=int(len(df)),
        atr_bars=int((df.get("atr", pd.Series([])) > 0).sum()),
        setups_long=int(df.get("enter_long", pd.Series([])).sum()),
        setups_short=0,
    )

    return summary, trades_df, equity_ser


# -------------------------
# Reporting
# -------------------------

def _plot_equity(equity: pd.Series, out: str):
    if equity.empty:
        return
    plt.figure()
    equity.plot()
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity (₹)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def _plot_drawdown(equity: pd.Series, out: str):
    if equity.empty:
        return
    roll_max = equity.cummax()
    dd = equity - roll_max
    plt.figure()
    dd.plot()
    plt.title("Drawdown (₹)")
    plt.xlabel("Trade #")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def save_reports(outdir, summary: Dict, trades_df: pd.DataFrame, equity_ser: pd.Series):
    outdir = pd.Path(outdir) if isinstance(outdir, str) else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # files
    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not equity_ser.empty:
        equity_ser.to_csv(outdir / "equity.csv", header=["equity"], index_label="ts")

    if not trades_df.empty:
        trades_df.to_csv(outdir / "trades.csv", index=False)

    # charts
    _plot_equity(equity_ser, outdir / "equity_curve.png")
    _plot_drawdown(equity_ser, outdir / "drawdown.png")

    # simple markdown
    lines = [
        "# Backtest Report",
        "",
        f"**Trades:** {summary.get('n_trades',0)}",
        f"**Win-rate:** {summary.get('win_rate',0)}%",
        f"**ROI:** {summary.get('roi_pct',0)}%",
        f"**PF:** {summary.get('profit_factor',0)}",
        f"**R:R:** {summary.get('rr',0)}",
        f"**Max DD:** {summary.get('max_dd_pct',0)}%",
        "",
        "Charts: `equity_curve.png`, `drawdown.png`",
    ]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
