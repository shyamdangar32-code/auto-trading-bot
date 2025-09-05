# bot/backtest.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time as dtime
from typing import Dict, Tuple

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless on CI
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))

def _within_session(ts: pd.Timestamp, sess_start: str, sess_end: str) -> bool:
    """
    Pandas 2.x: pd.Timestamp(hour=9, minute=15) જેવી કન્સ્ટ્રક્ટર પર ઇશ્યુ,
    એટલે compare માટે datetime.time() વાપરીએ છીએ.
    """
    tt = ts.tz_convert(None).time() if getattr(ts, "tzinfo", None) else ts.time()
    s = _parse_hhmm(sess_start)
    e = _parse_hhmm(sess_end)
    return (tt >= s) and (tt <= e)


@dataclass
class Position:
    side: str = "long"
    entry_px: float = 0.0
    sl_px: float = 0.0
    tp_px: float = 0.0
    qty: int = 0
    entry_ts: pd.Timestamp | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def _plot_equity(equity: pd.Series, out: Path):
    if equity is None or equity.empty:
        return
    plt.figure()
    equity.plot()
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity (₹)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def _plot_drawdown(equity: pd.Series, out: Path):
    if equity is None or equity.empty:
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


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, cfg: dict, use_block: str = "backtest_loose") -> Tuple[Dict, pd.DataFrame, pd.Series]:
    """
    Very simple long-only engine using SL/TP in ATR multiples.
    df MUST already include strategy columns like: enter_long, atr, High, Low, Close …

    Returns:
        summary (dict), trades_df (DataFrame), equity_ser (Series)
    """
    df = df.copy().sort_index()

    # parameters (fallbacks kept compatible with older configs)
    bcfg = (cfg.get("backtest") or {})
    filters = (bcfg.get("filters") or {})

    sess_s   = filters.get("session_start", bcfg.get("session_start", "09:20"))
    sess_e   = filters.get("session_end",   bcfg.get("session_end",   "15:20"))
    stop_mult = float((bcfg.get("exits") or {}).get("stop_atr_mult", cfg.get("stop_atr_mult", 1.0)))
    take_mult = float((bcfg.get("exits") or {}).get("take_atr_mult", cfg.get("take_atr_mult", 1.3)))

    capital      = float(cfg.get("capital_rs", 100000.0))
    default_qty  = int(cfg.get("order_qty", 1))

    # state
    pos: Position | None = None
    cash = capital
    eq_curve: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []

    for ts, row in df.iterrows():
        # session filter
        if not _within_session(ts, sess_s, sess_e):
            # square-off at session end
            if pos is not None:
                pnl = (float(row["Close"]) - pos.entry_px) * pos.qty
                cash += pnl
                trades.append(dict(
                    entry_ts=pos.entry_ts, exit_ts=ts, side="long",
                    entry=pos.entry_px, exit=float(row["Close"]), qty=pos.qty, pnl=pnl, reason="EOD"
                ))
                pos = None
            eq_curve.append((ts, cash))
            continue

        # ── Entry ────────────────────────────────────────────────────────────
        if pos is None and bool(row.get("enter_long", False)):
            entry = float(row["Close"])
            atr   = float(row.get("atr", 0.0))
            sl    = entry - stop_mult * atr
            tp    = entry + take_mult * atr

            # SAFE position size extraction (fixes IntCastingNaNError)
            qty_val = row.get("pos_size", None)
            try:
                if qty_val is not None and pd.notna(qty_val):
                    qty = int(float(qty_val))
                    if not np.isfinite(qty) or qty <= 0:
                        qty = default_qty
                else:
                    qty = default_qty
            except Exception:
                qty = default_qty

            pos = Position(side="long", entry_px=entry, sl_px=sl, tp_px=tp, qty=qty, entry_ts=ts)

        # ── Manage open position ─────────────────────────────────────────────
        if pos is not None:
            low  = float(row.get("Low", row["Close"]))
            high = float(row.get("High", row["Close"]))
            exit_px = None
            reason = None
            if low <= pos.sl_px:
                exit_px = pos.sl_px; reason = "SL"
            elif high >= pos.tp_px:
                exit_px = pos.tp_px; reason = "TP"

            # Optional soft exit hint (from strategy)
            if exit_px is None and bool(row.get("exit_long_hint", False)):
                exit_px = float(row["Close"]); reason = "HINT"

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
    if pos is not None and len(df):
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

    # equity
    equity_ser = pd.Series({ts: val for ts, val in eq_curve}).sort_index()
    base = capital
    ret = (equity_ser.iloc[-1] - base) / base if len(equity_ser) else 0.0

    # trades dataframe & metrics
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        win = rr = pf = 0.0
    else:
        wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
        losses = trades_df.loc[trades_df["pnl"] < 0, "pnl"]
        win = float((trades_df["pnl"] > 0).mean() * 100.0)
        rr  = (wins.mean() / abs(losses.mean())) if (len(wins) and len(losses)) else 0.0
        pf  = (wins.sum() / abs(losses.sum())) if (len(wins) and len(losses)) else 0.0

    roll_max = equity_ser.cummax() if len(equity_ser) else pd.Series(dtype=float)
    dd = (equity_ser - roll_max) if len(equity_ser) else pd.Series(dtype=float)
    max_dd = float((dd.min() / base) * 100) if len(dd) else 0.0
    time_dd_bars = int((dd == dd.min()).sum()) if len(dd) else 0

    summary = dict(
        n_trades=int(len(trades_df)),
        win_rate=round(win, 2),
        roi_pct=round(ret * 100, 2),
        profit_factor=round(pf, 2),
        rr=round(rr, 2),
        sharpe_ratio=0.0,            # placeholder
        max_dd_pct=round(max_dd, 2),
        time_dd_bars=time_dd_bars,
        n_bars=int(len(df)),
        atr_bars=int((df.get("atr", pd.Series(dtype=float)) > 0).sum()),
        setups_long=int(df.get("enter_long", pd.Series(dtype=float)).sum()),
        setups_short=0,
        profile=use_block.replace("backtest_", "").strip(),
    )

    return summary, trades_df, equity_ser


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def save_reports(outdir: str | Path,
                 summary: Dict,
                 trades_df: pd.DataFrame,
                 equity_ser: pd.Series) -> None:
    """
    Write metrics.json, trades.csv, equity.csv and charts into outdir.
    """
    out = Path(outdir) if isinstance(outdir, (str, Path)) else Path(".")
    out.mkdir(parents=True, exist_ok=True)

    # files
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if equity_ser is not None and not equity_ser.empty:
        equity_ser.to_csv(out / "equity.csv", header=["equity"], index_label="ts")

    if trades_df is not None and not trades_df.empty:
        trades_df.to_csv(out / "trades.csv", index=False)

    # charts
    _plot_equity(equity_ser, out / "equity_curve.png")
    _plot_drawdown(equity_ser, out / "drawdown.png")

    # quick markdown
    lines = [
        "# Backtest Report",
        "",
        f"**Profile:** {summary.get('profile','')}",
        f"**Trades:** {summary.get('n_trades',0)}",
        f"**Win-rate:** {summary.get('win_rate',0)}%",
        f"**ROI:** {summary.get('roi_pct',0)}%",
        f"**PF:** {summary.get('profit_factor',0)}",
        f"**R:R:** {summary.get('rr',0)}",
        f"**Max DD:** {summary.get('max_dd_pct',0)}%",
        "",
        "Charts: `equity_curve.png`, `drawdown.png`",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
