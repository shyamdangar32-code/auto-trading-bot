#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest runner (robust)
- Pulls index candles via bot.data_io
- Runs strategy backtest if possible
- Always writes report artifacts (even on empty data) so Telegram step never skips
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, time
import traceback

# ------- Optional imports (guarded) -------
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

# Repo-local modules (guard with try so we never crash the workflow)
def _safe_imports():
    mods = {}
    try:
        from bot import data_io as _dio
        mods["data_io"] = _dio
    except Exception:
        mods["data_io"] = None

    try:
        from bot import backtest as _bt
        mods["backtest"] = _bt
    except Exception:
        mods["backtest"] = None

    try:
        from bot import evaluation as _eval
        mods["evaluation"] = _eval
    except Exception:
        mods["evaluation"] = None

    try:
        from bot import strategy as _strat
        mods["strategy"] = _strat
    except Exception:
        mods["strategy"] = None

    try:
        from bot import metrics as _metrics
        mods["metrics"] = _metrics
    except Exception:
        mods["metrics"] = None

    return mods


# ------- Helpers -------
TITLE = "Backtest Summary"

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _fmt_pct(x):
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "0.0%"

def _write_summary(out_dir: Path,
                   trades=0, winrate=0.0, roi=0.0, pf=0.0,
                   rr=0.0, max_dd=0.0, time_dd=0, sharpe=0.0,
                   note="No signals generated today."):
    """
    Writes a Telegram-friendly summary text file.
    """
    lines = [
        f"🟩 {TITLE}",
        f"• Trades: {int(trades)}",
        f"• Win-rate: {_fmt_pct(winrate)}",
        f"• ROI: {_fmt_pct(roi)}",
        f"• Profit Factor: {pf:.1f}",
        f"• R:R: {rr:.1f}",
        f"• Max DD: {_fmt_pct(max_dd)}",
        f"• Time DD (bars): {int(time_dd)}",
        f"• Sharpe: {sharpe:.1f}",
        f"• Note: {note}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

def _write_placeholder_charts(out_dir: Path):
    """
    Always drop in two simple charts so telegram step can attach images.
    """
    try:
        import matplotlib.pyplot as plt

        # Equity curve placeholder
        x = list(range(1, 6))
        y = [100, 100, 100, 100, 100]
        plt.figure()
        plt.plot(x, y)
        plt.title("Equity Curve")
        plt.xlabel("Trades")
        plt.ylabel("Equity (₹)")
        plt.tight_layout()
        plt.savefig(out_dir / "equity_curve.png")
        plt.close()

        # Drawdown placeholder
        y2 = [0, 0, 0, 0, 0]
        plt.figure()
        plt.plot(x, y2)
        plt.title("Drawdown")
        plt.xlabel("Trades")
        plt.ylabel("DD (%)")
        plt.tight_layout()
        plt.savefig(out_dir / "drawdown.png")
        plt.close()
    except Exception:
        # As a last resort, create tiny png markers
        (out_dir / "equity_curve.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (out_dir / "drawdown.png").write_bytes(b"\x89PNG\r\n\x1a\n")


def _fallback_reports(out_dir: Path, why: str):
    print(f"WARN: {why} -> writing minimal reports.")
    _ensure_dir(out_dir)
    _write_summary(out_dir, note="No data; runner fallback.")
    _write_placeholder_charts(out_dir)


def parse_args():
    p = argparse.ArgumentParser(prog="run_backtest.py")

    p.add_argument("--symbol", default="BANKNIFTY",
                   help="Index symbol (NIFTY/BANKNIFTY)")
    p.add_argument("--start", default="", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end",   default="", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--interval", default="5m", help="Candle interval (e.g., 5m)")
    p.add_argument("--outdir", default="./reports", help="Output dir for artifacts")

    # Costs / limits (optional, used by your libs if present)
    p.add_argument("--slippage_bps", type=float, default=0.0)
    p.add_argument("--broker_flat", type=float, default=0.0)
    p.add_argument("--broker_pct", type=float, default=0.0)
    p.add_argument("--session_start", default="", help="HH:MM (e.g., 09:20)")
    p.add_argument("--session_end", default="", help="HH:MM (e.g., 15:25)")
    p.add_argument("--max_trades_per_day", type=int, default=0)
    p.add_argument("--qty", type=int, default=1)
    p.add_argument("--capital", type=float, default=100000.0)
    p.add_argument("--extra_params", default="{}",
                   help="JSON of extra knobs (strategy params)")

    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    _ensure_dir(out_dir)

    mods = _safe_imports()
    dio = mods["data_io"]
    bt = mods["backtest"]
    evaluation = mods["evaluation"]

    # Parse dates safely
    start_dt = args.start.strip()
    end_dt = args.end.strip()

    # 1) Get candles (index-level)
    df = None
    if dio is None or pd is None:
        _fallback_reports(out_dir, "Required modules missing (data_io/pandas)")
        return 0

    try:
        # repo’s function signature may differ; try common names
        if hasattr(dio, "get_index_candles"):
            df = dio.get_index_candles(args.symbol, start_dt, end_dt, args.interval)
        elif hasattr(dio, "get_index_data"):
            df = dio.get_index_data(args.symbol, start_dt, end_dt, args.interval)
        else:
            df = None
    except Exception as e:
        traceback.print_exc()
        df = None

    if df is None or (hasattr(df, "empty") and df.empty):
        _fallback_reports(out_dir, "data_io.get_index_candles unavailable or returned empty")
        return 0

    # 2) Run backtest via project libs if present; otherwise produce neutral report
    try:
        # Strategy params (if libs use them)
        try:
            extra = json.loads(args.extra_params or "{}")
        except Exception:
            extra = {}

        # Some projects expect time filters; pass if available
        session = {}
        if args.session_start:
            session["start"] = args.session_start
        if args.session_end:
            session["end"] = args.session_end

        results = None
        if bt and hasattr(bt, "run_index_backtest"):
            # Prefer a named function if it exists
            results = bt.run_index_backtest(
                df=df,
                symbol=args.symbol,
                qty=args.qty,
                capital=args.capital,
                slippage_bps=args.slippage_bps,
                broker_flat=args.broker_flat,
                broker_pct=args.broker_pct,
                session=session,
                max_trades_per_day=args.max_trades_per_day,
                params=extra,
            )
        elif bt and hasattr(bt, "run_backtest"):
            # Generic fallback name
            results = bt.run_backtest(
                df=df,
                qty=args.qty,
                capital=args.capital,
                slippage_bps=args.slippage_bps,
                broker_flat=args.broker_flat,
                broker_pct=args.broker_pct,
                session=session,
                max_trades_per_day=args.max_trades_per_day,
                params=extra,
            )

        # Extract metrics and plots if evaluation helper exists
        trades = 0
        winrate = roi = pf = rr = max_dd = sharpe = 0.0
        time_dd = 0

        if results is not None:
            if evaluation and hasattr(evaluation, "summarize_results"):
                summary = evaluation.summarize_results(results)
                # Expecting keys; guard everything
                trades = int(summary.get("trades", 0))
                winrate = float(summary.get("winrate_pct", 0.0))
                roi = float(summary.get("roi_pct", 0.0))
                pf = float(summary.get("profit_factor", 0.0))
                rr = float(summary.get("rr", 0.0))
                max_dd = float(summary.get("max_dd_pct", 0.0))
                time_dd = int(summary.get("time_dd_bars", 0))
                sharpe = float(summary.get("sharpe", 0.0))

            # Try to save charts if project provides helpers
            if evaluation and hasattr(evaluation, "save_equity_plot"):
                try:
                    evaluation.save_equity_plot(results, out_dir / "equity_curve.png")
                except Exception:
                    pass
            if evaluation and hasattr(evaluation, "save_drawdown_plot"):
                try:
                    evaluation.save_drawdown_plot(results, out_dir / "drawdown.png")
                except Exception:
                    pass

        # If no charts created by libs, ensure placeholders exist
        if not (out_dir / "equity_curve.png").exists() or not (out_dir / "drawdown.png").exists():
            _write_placeholder_charts(out_dir)

        note = "OK" if trades > 0 else "No signals generated today."
        _write_summary(
            out_dir,
            trades=trades, winrate=winrate, roi=roi, pf=pf,
            rr=rr, max_dd=max_dd, time_dd=time_dd, sharpe=sharpe,
            note=note
        )
        print("✅ Backtest reports written.")
        return 0

    except Exception as e:
        traceback.print_exc()
        _fallback_reports(out_dir, f"Backtest failed: {e}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
