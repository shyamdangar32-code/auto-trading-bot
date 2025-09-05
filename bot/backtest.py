# bot/backtest.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time as dtime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# charts (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))

def _within_session(ts: pd.Timestamp, sess_start: str, sess_end: str) -> bool:
    """
    Pandas 2.x માં સીધું pd.Timestamp(hour=..., minute=...) કરી compare કરવાથી
    સમસ્યા આવતી હતી. તેથી datetime.time દ્વારા compare કરીએ છીએ.
    """
    tt = ts.tz_convert(None).time() if getattr(ts, "tzinfo", None) else ts.time()
    s = _parse_hhmm(sess_start)
    e = _parse_hhmm(sess_end)
    return (tt >= s) and (tt <= e)


@dataclass
class Position:
    side: str = "long"                     # only long used here
    entry_px: float = 0.0
    sl_px: float = 0.0
    tp_px: float = 0.0
    qty: int = 0
    entry_ts: Optional[pd.Timestamp] = None


# -----------------------------------------------------------------------------
# Core backtest engine
# -----------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, cfg: dict, use_block: str = "backtest_loose") -> Tuple[Dict, pd.DataFrame, pd.Series]:
    """
    Very simple long-only engine with SL/TP. Exits at SL/TP or end-of-session.
    `df` must be time-indexed and should already contain signals produced by strategy:
      - enter_long (bool)
      - atr (float)  [optional if sl_px/tp_px already provided]
      - sl_px / tp_px / pos_size  [optional; engine will fall back to ATR multiples and default qty]
    Returns: (summary: dict, trades_df: DataFrame, equity_ser: Series)
    """
    df = df.copy().sort_index()

    # parameters / defaults
    bcfg = (cfg.get("backtest") or {})
    sess_s = (bcfg.get("filters", {}) or {}).get("session_start", bcfg.get("session_start", "09:20"))
    sess_e = (bcfg.get("filters", {}) or {}).get("session_end",   bcfg.get("session_end",   "15:20"))

    stop_mult = float(((bcfg.get("exits") or {}).get("stop_atr_mult", cfg.get("stop_atr_mult", 1.0))))
    take_mult = float(((bcfg.get("exits") or {}).get("take_atr_mult", cfg.get("take_atr_mult", 1.3))))

    capital = float(cfg.get("capital_rs", 100000.0))
    default_qty = int(cfg.get("order_qty", 1))

    # local state
    pos: Optional[Position] = None
    cash = capital
    eq_curve = []
    trades = []

    for ts, row in df.iterrows():
        # session window filter
        if not _within_session(ts, sess_s, sess_e):
            # square-off at session boundary if position is open
            if pos is not None:
                pnl = (float(row["Close"]) - pos.entry_px) * pos.qty
                cash += pnl
                trades.append(dict(
                    entry_ts=pos.entry_ts, exit_ts=ts, side="long",
                    entry=pos.entry_px, exit=float(row["Close"]),
                    qty=pos.qty, pnl=pnl, reason="EOD"
                ))
                pos = None
            eq_curve.append((ts, cash))
            continue

        # entry
        if pos is None and bool(row.get("enter_long", False)):
            entry = float(row["Close"])
            atr   = float(row.get("atr", 0.0))

            # Prefer signals-provided stops/targets/quantity if present
            sl = float(row["sl_px"]) if "sl_px" in row and pd.notna(row["sl_px"]) else (entry - stop_mult * atr)
            tp = float(row["tp_px"]) if "tp_px" in row and pd.notna(row["tp_px"]) else (entry + take_mult * atr)
            qty = int(row["pos_size"]) if "pos_size" in row and pd.notna(row["pos_size"]) and int(row["pos_size"]) > 0 else default_qty

            pos = Position(side="long", entry_px=entry, sl_px=sl, tp_px=tp, qty=qty, entry_ts=ts)

        # manage open position
        if pos is not None:
            low = float(row.get("Low", row["Close"]))
            high = float(row.get("High", row["Close"]))

            exit_px = None
            reason = None

            # hard SL/TP
            if low <= pos.sl_px:
                exit_px, reason = pos.sl_px, "SL"
            elif high >= pos.tp_px:
                exit_px, reason = pos.tp_px, "TP"
            # optional early exit hint from strategy
            elif bool(row.get("exit_long_hint", False)):
                exit_px, reason = float(row["Close"]), "HINT"

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

    # equity series
    equity_ser = pd.Series({ts: val for ts, val in eq_curve}).sort_index()

    # -----------------------------------------------------------------------------
    # Metrics (guarded for empty trades)
    # -----------------------------------------------------------------------------
    trades_df = pd.DataFrame(trades)
    base = capital
    roi = (equity_ser.iloc[-1] - base) / base if len(equity_ser) else 0.0

    if not trades_df.empty and "pnl" in trades_df.columns:
        win_rate = float((trades_df["pnl"] > 0).mean() * 100.0)
        pos_pnl = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
        neg_pnl = trades_df.loc[trades_df["pnl"] < 0, "pnl"]
        rr = float(pos_pnl.mean() / abs(neg_pnl.mean())) if (len(pos_pnl) > 0 and len(neg_pnl) > 0) else 0.0
        pf = float(pos_pnl.sum() / abs(neg_pnl.sum())) if (pos_pnl.sum() != 0 and neg_pnl.sum() != 0) else 0.0
    else:
        win_rate = 0.0
        rr = 0.0
        pf = 0.0

    # drawdown
    if len(equity_ser):
        roll_max = equity_ser.cummax()
        dd = equity_ser - roll_max
        max_dd_pct = float((dd.min() / base) * 100.0)
        time_dd_bars = int((dd == dd.min()).sum())
    else:
        max_dd_pct = 0.0
        time_dd_bars = 0

    summary = dict(
        n_trades=int(len(trades_df)),
        win_rate=round(win_rate, 2),
        roi_pct=round(roi * 100.0, 2),
        profit_factor=round(pf, 2),
        rr=round(rr, 2),
        sharpe_ratio=0.0,                 # placeholder
        max_dd_pct=round(max_dd_pct, 2),
        time_dd_bars=time_dd_bars,
        n_bars=int(len(df)),
        atr_bars=int((df.get("atr", pd.Series(dtype=float)) > 0).sum()),
        setups_long=int(df.get("enter_long", pd.Series(dtype=int)).sum()),
        setups_short=0,
    )

    return summary, trades_df, equity_ser


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def _plot_equity(equity: pd.Series, out: Path):
    if equity.empty:
        return
    plt.figure()
    equity.plot()
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (₹)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def _plot_drawdown(equity: pd.Series, out: Path):
    if equity.empty:
        return
    roll_max = equity.cummax()
    dd = equity - roll_max
    plt.figure()
    dd.plot()
    plt.title("Drawdown (₹)")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def save_reports(outdir, summary: Dict, trades_df: pd.DataFrame, equity_ser: pd.Series):
    """
    Writes:
      - metrics.json
      - trades.csv (if any)
      - equity.csv (if any)
      - equity_curve.png / drawdown.png
      - report.md (quick human summary)
    """
    outdir = Path(outdir) if isinstance(outdir, (str, Path)) else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # metrics
    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # csvs
    if not equity_ser.empty:
        equity_ser.to_csv(outdir / "equity.csv", header=["equity"], index_label="ts")
    if not trades_df.empty:
        trades_df.to_csv(outdir / "trades.csv", index=False)

    # charts
    _plot_equity(equity_ser, outdir / "equity_curve.png")
    _plot_drawdown(equity_ser, outdir / "drawdown.png")

    # minimal markdown
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
