# bot/backtest.py
from __future__ import annotations
import os, math, json
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .strategy import prepare_signals
from .metrics import compute_metrics


@dataclass
class RiskConfig:
    stop_atr_mult: float = 1.0
    take_atr_mult: float = 1.3
    trail_type: str = "atr"      # "atr" | "off"
    trail_start_atr: float = 0.5
    trail_atr_mult: float = 1.0
    step_bars: int = 3
    risk_pct: float = 0.005      # 0.5% equity risk per trade
    min_stop_pct: float = 0.002  # 0.2% absolute minimum stop


def _within_session(ts: pd.Timestamp, s="09:25", e="15:15") -> bool:
    t = ts.tz_localize(None).time()
    sh, sm = map(int, s.split(":"))
    eh, em = map(int, e.split(":"))
    return (t >= pd.Timestamp(hour=sh, minute=sm).time()) and (t <= pd.Timestamp(hour=eh, minute=em).time())


def _fees(price: float, qty: int, broker_flat: float, broker_pct: float, slippage_bps: float) -> float:
    # simple per-leg fee + slippage model (roundtrip applied by engine)
    brokerage = max(broker_flat, price * qty * broker_pct)
    slippage  = price * qty * (slippage_bps / 10000.0)
    return brokerage + slippage


def _size_position(equity: float, price: float, atr: float, risk: RiskConfig, lot_size: int = 1) -> int:
    stop_dist = max(risk.stop_atr_mult * atr, risk.min_stop_pct * price)
    if stop_dist <= 0:
        return 0
    risk_rs = equity * risk.risk_pct
    qty = int(math.floor((risk_rs / stop_dist) / lot_size)) * lot_size
    return max(qty, lot_size)


def run_backtest(prices: pd.DataFrame, cfg: Dict, profile: str = "loose", use_block: str | None = None
                 ) -> Tuple[Dict, pd.DataFrame, pd.Series]:
    """
    Core backtest loop (long-only).
    Returns (summary_dict, trades_df, equity_series)
    """
    bt = cfg.get("backtest", {}) or {}
    # session window (fallback to global filters if provided)
    sess_s = (bt.get("filters", {}) or {}).get("session_start", bt.get("session_start", "09:20"))
    sess_e = (bt.get("filters", {}) or {}).get("session_end",   bt.get("session_end",   "15:20"))

    # execution frictions
    slippage_bps = float(bt.get("slippage_bps", 2.0))
    broker_flat  = float(bt.get("brokerage_flat", 20.0))
    broker_pct   = float(bt.get("brokerage_pct", 0.0003))

    # risk/exits
    r = RiskConfig(
        stop_atr_mult=float((bt.get("exits", {}) or {}).get("stop_atr_mult", cfg.get("stop_atr_mult", 1.0))),
        take_atr_mult=float((bt.get("exits", {}) or {}).get("take_atr_mult", cfg.get("take_atr_mult", 1.3))),
        trail_type=((bt.get("exits", {}).get("trail", {}) or {}).get("type", "atr")).lower(),
        trail_start_atr=float((bt.get("exits", {}).get("trail", {}) or {}).get("atr_mult", cfg.get("trail_start_atr", 0.5))),
        trail_atr_mult=float((bt.get("exits", {}).get("trail", {}) or {}).get("atr_mult", cfg.get("trail_atr_mult", 1.0))),
        step_bars=int((bt.get("exits", {}).get("trail", {}) or {}).get("step_bars", 3)),
        risk_pct=float(bt.get("risk_pct", 0.005)),
        min_stop_pct=float(bt.get("min_stop_pct", 0.002)),
    )

    capital = float(cfg.get("capital_rs", 100000.0))
    order_qty_hint = int(cfg.get("order_qty", 1))

    # signals/indicators
    df = prepare_signals(prices, cfg, profile=profile)
    df = df[(~df.index.duplicated(keep="last"))].copy()

    equity = capital
    in_pos = False
    entry_px = sl = tp = trail = np.nan
    qty = 0
    entry_time = None

    trades = []

    step_ctr = 0
    for ts, row in df.iterrows():
        # session & data checks
        if not _within_session(ts, sess_s, sess_e):
            continue
        price = float(row["Close"])
        atr   = float(row["atr"]) if not np.isnan(row["atr"]) else 0.0

        # trailing refresh cadence
        step_ctr += 1

        if not in_pos:
            if row["enter_long"]:
                # position sizing
                size = _size_position(equity, price, atr, r, lot_size=max(1, order_qty_hint))
                if size <= 0:
                    continue

                # set SL/TP
                stop_dist = max(r.stop_atr_mult * atr, r.min_stop_pct * price)
                take_dist = r.take_atr_mult * atr

                entry_px = price
                sl = entry_px - stop_dist
                tp = entry_px + take_dist
                trail = entry_px  # start point; will move when in profit
                qty = size
                entry_time = ts
                in_pos = True

                # pay entry fees & slippage
                equity -= _fees(price, qty, broker_flat, broker_pct, slippage_bps)

        else:
            # manage long
            hit_sl = price <= sl
            hit_tp = price >= tp

            # trailing activation: when in profit by trail_start_atr * ATR
            if r.trail_type == "atr" and (price - entry_px) >= (r.trail_start_atr * atr):
                if step_ctr % max(1, r.step_bars) == 0:
                    # move trail to (price - trail_atr_mult*ATR) but never below previous trail
                    new_trail = price - r.trail_atr_mult * atr
                    trail = max(trail, new_trail)
                    sl = max(sl, trail)

            if hit_sl or hit_tp or row["exit_long_hint"]:
                exit_px = price
                pnl = (exit_px - entry_px) * qty
                equity += pnl
                # pay exit fees
                equity -= _fees(price, qty, broker_flat, broker_pct, slippage_bps)

                trades.append(dict(
                    entry_time=entry_time, exit_time=ts,
                    side="LONG", qty=int(qty),
                    entry=float(entry_px), exit=float(exit_px),
                    sl=float(sl), tp=float(tp),
                    pnl=float(pnl)
                ))

                # reset position
                in_pos = False
                entry_px = sl = tp = trail = np.nan
                qty = 0
                entry_time = None
                step_ctr = 0

    # close open trade at last price (neutral exit)
    if in_pos and len(df) > 0:
        last_ts = df.index[-1]
        price = float(df["Close"].iloc[-1])
        pnl = (price - entry_px) * qty
        equity += pnl
        equity -= _fees(price, qty, broker_flat, broker_pct, slippage_bps)
        trades.append(dict(
            entry_time=entry_time, exit_time=last_ts,
            side="LONG", qty=int(qty),
            entry=float(entry_px), exit=float(price),
            sl=float(sl), tp=float(tp),
            pnl=float(pnl)
        ))

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.Series(dtype=float)
    if not trades_df.empty:
        equity_curve = (capital + trades_df["pnl"].cumsum())
        equity_curve.index = pd.RangeIndex(start=1, stop=len(equity_curve)+1)

    # metrics
    summary = compute_metrics(trades_df, equity_curve, starting_capital=capital)
    return summary, trades_df, equity_curve


# ---------- Reporting helpers ----------
def _plot_equity_dd(equity: pd.Series, outdir: str):
    if equity is None or equity.empty:
        return
    # Equity
    plt.figure()
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve")
    plt.xlabel("Trade #"); plt.ylabel("Equity (₹)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "equity_curve.png"))
    plt.close()

    # Drawdown (trade-level)
    roll = equity.cummax()
    dd = equity - roll
    plt.figure()
    plt.plot(equity.index, dd.values)
    plt.title("Drawdown (₹)")
    plt.xlabel("Trade #"); plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "drawdown.png"))
    plt.close()


def save_reports(outdir: str, summary: Dict, trades: pd.DataFrame, equity: pd.Series, meta: Dict | None = None):
    os.makedirs(outdir, exist_ok=True)
    # CSVs
    if trades is not None:
        trades.to_csv(os.path.join(outdir, "trades.csv"), index=False)
    if equity is not None and not equity.empty:
        equity.to_csv(os.path.join(outdir, "equity.csv"), index=False, header=["equity"])
    # JSON metrics
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    # Plots
    _plot_equity_dd(equity, outdir)
    # Markdown mini-report
    lines = [
        "# Backtest Report",
        "",
        f"- Trades: {summary.get('n_trades',0)}",
        f"- Win-rate: {summary.get('win_rate',0)}%",
        f"- ROI: {summary.get('roi_pct',0)}%",
        f"- Profit Factor: {summary.get('profit_factor',0)}",
        f"- R:R: {summary.get('rr',0)}",
        f"- Sharpe: {summary.get('sharpe_ratio',0)}",
        f"- Max DD: {summary.get('max_dd_pct',0)}%  | Time DD: {summary.get('time_dd_bars',0)} bars",
    ]
    if meta:
        lines += ["", "## Meta", json.dumps(meta, indent=2)]
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
