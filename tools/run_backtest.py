#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-contained index-level backtester with:
- Session filter (start/end HH:MM)
- Slippage (bps) + Brokerage (flat or % of turnover)
- Max trades per day
- Simple EMA(20/50) + RSI(14) strategy (buy when EMA20>EMA50 and RSI>55, flat otherwise)
Outputs:
- reports/metrics.json
- reports/latest_signals.csv
- reports/equity_curve.png
- reports/drawdown.png
"""

import argparse
import json
import math
import os
from pathlib import Path
from datetime import datetime, timedelta, time, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Repo helper (data loader) ---
try:
    from data_io import get_index_candles  # expected in your repo
except Exception:
    get_index_candles = None


# ---------------- Utils ----------------
def to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    # GitHub runner is UTC; convert to IST for session filter & charts readability
    try:
        ist = timezone(timedelta(hours=5, minutes=30))
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        return ts.tz_convert(ist)
    except Exception:
        return ts


def within_session(ts: pd.Timestamp, session_start: str, session_end: str) -> bool:
    """Check if timestamp is within trading session [start, end)."""
    ts_ist = to_ist(ts)
    HH, MM = map(int, session_start.split(":"))
    s = time(HH, MM)
    HH2, MM2 = map(int, session_end.split(":"))
    e = time(HH2, MM2)
    t = ts_ist.time()
    return (t >= s) and (t < e)


def apply_costs(entry_price, exit_price, qty, slippage_bps, broker_flat, broker_pct):
    """
    Costs:
      - Slippage: both on entry & exit (bps of price).
      - Brokerage: max(flat, pct * turnover) per leg (entry + exit).
    """
    slip_entry = entry_price * (slippage_bps / 1e4)
    slip_exit = exit_price * (slippage_bps / 1e4)

    entry_exec = entry_price + slip_entry
    exit_exec = exit_price - slip_exit

    turnover_entry = entry_exec * qty
    turnover_exit = exit_exec * qty

    br_entry = max(broker_flat, broker_pct * turnover_entry)
    br_exit = max(broker_flat, broker_pct * turnover_exit)

    total_cost = br_entry + br_exit
    return entry_exec, exit_exec, total_cost


def compute_drawdown(series: pd.Series) -> pd.Series:
    peak = series.cummax()
    dd = (series - peak) / peak.replace(0, np.nan)
    return dd.fillna(0.0)


def sharpe_ratio(returns, rf=0.0):
    if len(returns) < 2:
        return 0.0
    ex = returns - rf
    std = ex.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return (ex.mean() / std) * np.sqrt(252)  # approx trading days


# -------------- Strategy ---------------
def generate_signals(df: pd.DataFrame, ema_fast=20, ema_slow=50, rsi_len=14, rsi_buy=55):
    """Simple index-level intraday bias: long-only when EMA fast > EMA slow and RSI > threshold."""
    close = df["close"]
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()

    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(rsi_len).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(rsi_len).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.fillna(50.0)

    long_bias = (ema_f > ema_s) & (rsi > rsi_buy)
    # enter when bias turns True, exit when bias turns False or session end
    sig = long_bias.astype(int).diff().fillna(0)
    entries = sig.eq(1)
    exits = sig.eq(-1)

    out = pd.DataFrame({
        "long_bias": long_bias,
        "entry_flag": entries,
        "exit_flag": exits,
    }, index=df.index)
    return out


# -------------- Backtest Engine --------------
def run_backtest(
    df,
    capital=100000.0,
    qty=1,
    slippage_bps=2.0,
    broker_flat=20.0,
    broker_pct=0.0003,  # 3 bps
    session_start="09:20",
    session_end="15:20",
    max_trades_per_day=6,
    plot_outdir="reports",
):
    if df is None or len(df) == 0:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "profit_factor": 0.0,
            "rr": 0.0,
            "max_dd_pct": 0.0,
            "time_dd_bars": 0,
            "sharpe": 0.0,
        }, pd.DataFrame(), pd.Series(dtype=float)

    df = df.copy()
    # Filter to trading session
    mask = [within_session(ts, session_start, session_end) for ts in df.index]
    df = df.loc[mask]
    if df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "profit_factor": 0.0,
            "rr": 0.0,
            "max_dd_pct": 0.0,
            "time_dd_bars": 0,
            "sharpe": 0.0,
        }, pd.DataFrame(), pd.Series(dtype=float)

    sig = generate_signals(df)

    # Build trades
    day_key = df.index.tz_convert("Asia/Kolkata").date if df.index.tz is not None else df.index.date
    day_series = pd.Series(day_key, index=df.index)

    open_pos = False
    entry_price = None
    trades = []
    trades_today = 0
    current_day = None

    for ts, row in df.iterrows():
        day = day_series.loc[ts]
        if current_day != day:
            current_day = day
            trades_today = 0

        # session end exit
        if open_pos:
            # if signal turns off or we hit max time (next candle after session end will be filtered already)
            if sig.loc[ts, "exit_flag"]:
                exit_price_raw = row["close"]
                ent, ex, cost = apply_costs(entry_price, exit_price_raw, qty, slippage_bps, broker_flat, broker_pct)
                pnl = (ex - ent) * qty - cost
                trades.append({"entry_time": entry_time, "exit_time": ts, "entry": ent, "exit": ex,
                               "qty": qty, "pnl": pnl})
                open_pos = False
                entry_price = None

        # entries
        if (not open_pos) and sig.loc[ts, "entry_flag"] and (trades_today < max_trades_per_day):
            entry_price = row["close"]
            entry_time = ts
            open_pos = True
            trades_today += 1

    # force close last open at last bar of day
    if open_pos:
        last_ts = df.index[-1]
        last_price = df["close"].iloc[-1]
        ent, ex, cost = apply_costs(entry_price, last_price, qty, slippage_bps, broker_flat, broker_pct)
        pnl = (ex - ent) * qty - cost
        trades.append({"entry_time": entry_time, "exit_time": last_ts, "entry": ent, "exit": ex,
                       "qty": qty, "pnl": pnl})
        open_pos = False

    trades_df = pd.DataFrame(trades)

    # Equity curve (bar-level; mark-to-market between trades just carries last equity)
    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = capital
    cum = capital
    pf_gross_win = 0.0
    pf_gross_loss = 0.0

    if not trades_df.empty:
        for _, tr in trades_df.iterrows():
            # snap equity at exit candle
            cum += tr["pnl"]
            if tr["pnl"] >= 0:
                pf_gross_win += tr["pnl"]
            else:
                pf_gross_loss += -tr["pnl"]
            if tr["exit_time"] in equity.index:
                equity.loc[tr["exit_time"]] = cum
        equity = equity.ffill().fillna(cum)
    else:
        equity = pd.Series(capital, index=df.index)

    returns = equity.pct_change().fillna(0.0)
    dd = compute_drawdown(equity)
    max_dd_pct = dd.min() * 100.0
    time_dd_bars = int((dd < 0).astype(int).groupby((dd >= 0).astype(int).cumsum()).sum().max() or 0)

    wins = (trades_df["pnl"] > 0).sum() if not trades_df.empty else 0
    total = len(trades_df)
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    roi = ((equity.iloc[-1] - capital) / capital) * 100.0
    profit_factor = (pf_gross_win / pf_gross_loss) if pf_gross_loss > 0 else (pf_gross_win > 0 and 99.0 or 0.0)

    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() if total else 0.0
    avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].abs().mean() if total else 0.0
    rr = (avg_win / avg_loss) if avg_loss and not math.isnan(avg_loss) else 0.0
    shrp = float(sharpe_ratio(returns))

    metrics = {
        "trades": int(total),
        "win_rate": round(win_rate, 2),
        "roi": round(roi, 2),
        "profit_factor": round(float(profit_factor), 2) if isinstance(profit_factor, (int, float)) else 0.0,
        "rr": round(float(rr), 2) if isinstance(rr, (int, float)) else 0.0,
        "max_dd_pct": round(float(max_dd_pct), 2),
        "time_dd_bars": int(time_dd_bars),
        "sharpe": round(shrp, 2),
    }
    return metrics, trades_df, equity


def save_reports(outdir, metrics, trades_df, equity):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # metrics.json
    with open(Path(outdir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # latest_signals.csv (trades)
    trades_path = Path(outdir) / "latest_signals.csv"
    if trades_df is None or trades_df.empty:
        pd.DataFrame(columns=["entry_time", "exit_time", "entry", "exit", "qty", "pnl"]).to_csv(trades_path, index=False)
    else:
        tdf = trades_df.copy()
        tdf["entry_time"] = tdf["entry_time"].astype(str)
        tdf["exit_time"] = tdf["exit_time"].astype(str)
        tdf.to_csv(trades_path, index=False)

    # equity curve
    plt.figure()
    equity.plot()
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "equity_curve.png")
    plt.close()

    # drawdown
    plt.figure()
    dd = compute_drawdown(equity) * 100.0
    dd.plot()
    plt.title("Drawdown (%)")
    plt.xlabel("Time")
    plt.ylabel("DD %")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "drawdown.png")
    plt.close()


def load_cfg(cfg_path):
    cfg = {}
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Index-level backtester (with costs + session filter)")
    parser.add_argument("--symbol", type=str, default=os.environ.get("SYMBOL", "NIFTY"))
    parser.add_argument("--start", type=str, default=os.environ.get("START", ""))   # YYYY-MM-DD
    parser.add_argument("--end", type=str, default=os.environ.get("END", ""))       # YYYY-MM-DD
    parser.add_argument("--interval", type=str, default=os.environ.get("INTERVAL", "5m"))
    parser.add_argument("--outdir", type=str, default="./reports")

    # Overridables (else read from config.yaml)
    parser.add_argument("--slippage_bps", type=float, default=None)
    parser.add_argument("--broker_flat", type=float, default=None)
    parser.add_argument("--broker_pct", type=float, default=None)
    parser.add_argument("--session_start", type=str, default=None)
    parser.add_argument("--session_end", type=str, default=None)
    parser.add_argument("--max_trades_per_day", type=int, default=None)
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--capital", type=float, default=100000.0)

    args = parser.parse_args()

    cfg = load_cfg("config.yaml")
    bt = (cfg.get("backtest") or {})

    slippage_bps = args.slippage_bps if args.slippage_bps is not None else float(bt.get("slippage_bps", 2.0))
    broker_flat = args.broker_flat if args.broker_flat is not None else float(bt.get("brokerage_flat", 20.0))
    broker_pct  = args.broker_pct  if args.broker_pct  is not None else float(bt.get("brokerage_pct", 0.0003))
    session_start = args.session_start or bt.get("session_start", "09:20")
    session_end   = args.session_end   or bt.get("session_end", "15:20")
    max_tpd = args.max_trades_per_day if args.max_trades_per_day is not None else int(bt.get("max_trades_per_day", 6))

    # Dates
    if not args.end:
        end = datetime.utcnow().date()
    else:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if not args.start:
        start = end - timedelta(days=30)
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()

    # Load candles
    df = None
    if get_index_candles is not None:
        try:
            df = get_index_candles(
                symbol=args.symbol,
                start=str(start),
                end=str(end),
                interval=args.interval,
            )
            if isinstance(df, pd.DataFrame) and "close" in df.columns:
                df = df.sort_index()
            else:
                df = None
        except Exception as e:
            print(f"ERROR loading candles: {e}")

    if df is None or df.empty:
        print("WARN: data_io.get_index_candles unavailable or returned empty → writing minimal reports.")
        metrics = {
            "trades": 0, "win_rate": 0.0, "roi": 0.0, "profit_factor": 0.0,
            "rr": 0.0, "max_dd_pct": 0.0, "time_dd_bars": 0, "sharpe": 0.0
        }
        save_reports(args.outdir, metrics, pd.DataFrame(), pd.Series(dtype=float))
        return

    metrics, trades_df, equity = run_backtest(
        df=df,
        capital=args.capital,
        qty=args.qty,
        slippage_bps=slippage_bps,
        broker_flat=broker_flat,
        broker_pct=broker_pct,
        session_start=session_start,
        session_end=session_end,
        max_trades_per_day=max_tpd,
        plot_outdir=args.outdir,
    )
    save_reports(args.outdir, metrics, trades_df, equity)
    print("Backtest done. Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
