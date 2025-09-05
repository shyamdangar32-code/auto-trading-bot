# bot/backtest.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time as dtime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))

def _within_session(ts: pd.Timestamp, sess_start: str, sess_end: str) -> bool:
    tt = ts.tz_convert(None).time() if getattr(ts, "tzinfo", None) else ts.time()
    s = _parse_hhmm(sess_start)
    e = _parse_hhmm(sess_end)
    return (tt >= s) and (tt <= e)

def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


@dataclass
class Position:
    side: str = "long"
    entry_px: float = 0.0
    sl_px: float = 0.0
    tp_px: float = 0.0
    qty: int = 0
    entry_ts: pd.Timestamp | None = None


# -------------------------
# Engine
# -------------------------

def run_backtest(prices: pd.DataFrame, cfg: dict, use_block: str = "backtest_loose"):
    """
    Long-only toy engine with ATR SL/TP.
    Now includes:
      • Risk % based sizing at entry
      • Cash/notional guardrails (no leverage; cash can't go < 0)
      • Commission + slippage
      • Daily-trade cap
    Returns: (summary: dict, trades_df: DataFrame, equity_ser: Series)
    """

    if prices is None or prices.empty:
        return dict(n_trades=0, win_rate=0.0, roi_pct=0.0, profit_factor=0.0,
                    rr=0.0, sharpe_ratio=0.0, max_dd_pct=0.0,
                    time_dd_bars=0, n_bars=0, atr_bars=0,
                    setups_long=0, setups_short=0), pd.DataFrame(), pd.Series(dtype="float64")

    df = prices.copy().sort_index()

    # ---- parameters (with sensible defaults) ----
    bcfg = (cfg.get("backtest") or {})
    filters = (bcfg.get("filters") or {})

    sess_s = filters.get("session_start", bcfg.get("session_start", "09:20"))
    sess_e = filters.get("session_end",   bcfg.get("session_end",   "15:20"))

    stop_mult = float((bcfg.get("exits") or {}).get("stop_atr_mult", cfg.get("stop_atr_mult", 1.0)))
    take_mult = float((bcfg.get("exits") or {}).get("take_atr_mult", cfg.get("take_atr_mult", 1.3)))

    # Risk & costs
    capital = float(cfg.get("capital_rs", 100000.0))
    risk_pct = float(bcfg.get("risk_per_trade", cfg.get("risk_per_trade", 0.005)))  # 0.5% default
    max_pos_pct = float(bcfg.get("max_position_notional_pct", 0.99))                # max notional vs cash
    max_daily_trades = int(bcfg.get("max_daily_trades", 50))
    point_value = float(bcfg.get("point_value", cfg.get("point_value", 1.0)))       # ₹ per 1 price-point
    commission_per_trade = float(bcfg.get("commission_rs", 0.0))                    # both sides together
    slippage_points = float(bcfg.get("slippage_points", 0.0))                       # per fill, points

    # columns required from strategy.prepare_signals()
    # fill safe defaults if missing
    df["enter_long"] = df.get("enter_long", pd.Series(False, index=df.index)).fillna(False)
    atr = df.get("atr", pd.Series(0.0, index=df.index)).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["atr"] = atr
    low  = _safe_float(df["Low"].iloc[0], default=0.0) if "Low" in df.columns else 0.0
    high = _safe_float(df["High"].iloc[0], default=0.0) if "High" in df.columns else 0.0
    # ensure OHLC present
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"prices is missing required column: {col}")

    # ---- local state ----
    pos: Position | None = None
    cash = capital
    eq_curve: list[Tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []
    daily_counts: dict[Tuple[int, int, int], int] = {}

    def _daily_key(ts: pd.Timestamp) -> Tuple[int, int, int]:
        dt = ts.to_pydatetime()
        return (dt.year, dt.month, dt.day)

    for ts, row in df.iterrows():
        # session & daily cap
        if not _within_session(ts, sess_s, sess_e):
            # square-off outside session
            if pos is not None:
                exit_px = float(row["Close"]) - slippage_points  # market-out with slippage
                pnl = (exit_px - pos.entry_px) * pos.qty * point_value - commission_per_trade
                cash = max(0.0, cash + pnl)
                trades.append(dict(entry_ts=pos.entry_ts, exit_ts=ts, side="long",
                                   entry=pos.entry_px, exit=exit_px, qty=pos.qty, pnl=pnl, reason="EOD"))
                pos = None
            eq_curve.append((ts, cash))
            continue

        # daily trade count check
        dkey = _daily_key(ts)
        if dkey not in daily_counts:
            daily_counts[dkey] = 0

        # Entries
        if pos is None and bool(row["enter_long"]) and daily_counts[dkey] < max_daily_trades:
            price = float(row["Close"]) + slippage_points
            atr_val = max(0.01, float(row.get("atr", 0.0)))  # avoid 0
            stop_distance = max(0.5, atr_val * stop_mult)    # at least 0.5 point
            risk_rupees = max(1.0, capital * risk_pct)

            # Qty by risk-per-trade
            risk_per_unit = stop_distance * point_value
            qty = int(risk_rupees // risk_per_unit) if risk_per_unit > 0 else 0

            # notional cap (no leverage)
            if qty > 0:
                max_qty_by_cash = int((cash * max_pos_pct) // (price * point_value))
                qty = max(0, min(qty, max_qty_by_cash))

            if qty > 0:
                sl = price - stop_distance
                tp = price + max(stop_distance, atr_val * take_mult)
                pos = Position(side="long", entry_px=price, sl_px=sl, tp_px=tp, qty=qty, entry_ts=ts)
                # pay entry costs immediately
                cash = max(0.0, cash - commission_per_trade)
                daily_counts[dkey] += 1

        # Manage open
        if pos is not None:
            hi = float(row.get("High", row["Close"]))
            lo = float(row.get("Low",  row["Close"]))
            exit_px: float | None = None
            reason = None

            if lo <= pos.sl_px:
                exit_px = pos.sl_px - slippage_points
                reason = "SL"
            elif hi >= pos.tp_px:
                exit_px = pos.tp_px - slippage_points
                reason = "TP"

            if exit_px is not None:
                pnl = (exit_px - pos.entry_px) * pos.qty * point_value - commission_per_trade
                cash = max(0.0, cash + pnl)
                trades.append(dict(entry_ts=pos.entry_ts, exit_ts=ts, side="long",
                                   entry=pos.entry_px, exit=exit_px, qty=pos.qty, pnl=pnl, reason=reason))
                pos = None

        eq_curve.append((ts, cash))

    # finalize at last bar
    if pos is not None:
        last_ts = df.index[-1]
        last_px = float(df.iloc[-1]["Close"]) - slippage_points
        pnl = (last_px - pos.entry_px) * pos.qty * point_value - commission_per_trade
        cash = max(0.0, cash + pnl)
        trades.append(dict(entry_ts=pos.entry_ts, exit_ts=last_ts, side="long",
                           entry=pos.entry_px, exit=last_px, qty=pos.qty, pnl=pnl, reason="EOD"))
        pos = None
        eq_curve[-1] = (last_ts, cash)

    equity_ser = pd.Series({ts: val for ts, val in eq_curve}).sort_index()

    base = capital
    ret = (equity_ser.iloc[-1] - base) / base if len(equity_ser) else 0.0

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        win = rr = pf = 0.0
    else:
        win = float((trades_df["pnl"] > 0).mean() * 100.0)
        if (trades_df["pnl"] > 0).any() and (trades_df["pnl"] < 0).any():
            rr = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
                       / abs(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()))
            pf = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
                       / abs(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()))
        else:
            rr = pf = 0.0

    # drawdown on cash curve
    roll_max = equity_ser.cummax() if len(equity_ser) else pd.Series(dtype="float64")
    dd = equity_ser - roll_max if len(equity_ser) else pd.Series(dtype="float64")
    max_dd = float((dd.min() / base) * 100.0) if len(dd) else 0.0
    time_dd_bars = int((dd == dd.min()).sum()) if len(dd) else 0

    summary = dict(
        n_trades=int(len(trades_df)),
        win_rate=round(win, 2),
        roi_pct=round(ret * 100.0, 2),
        profit_factor=round(pf, 2),
        rr=round(rr, 2),
        sharpe_ratio=0.0,        # placeholder
        max_dd_pct=round(max_dd, 2),
        time_dd_bars=time_dd_bars,
        n_bars=int(len(df)),
        atr_bars=int((df["atr"] > 0).sum()),
        setups_long=int(df["enter_long"].sum()),
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
    from pathlib import Path as _Path
    outdir = _Path(outdir) if isinstance(outdir, str) else outdir
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not equity_ser.empty:
        equity_ser.to_csv(outdir / "equity.csv", header=["equity"], index_label="ts")
    if not trades_df.empty:
        trades_df.to_csv(outdir / "trades.csv", index=False)

    _plot_equity(equity_ser, outdir / "equity_curve.png")
    _plot_drawdown(equity_ser, outdir / "drawdown.png")

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
