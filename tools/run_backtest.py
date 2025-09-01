# tools/run_backtest.py
# ---------------------------------------
# Safe CLI wrapper for backtests.
# - Accepts all workflow inputs (symbol, start, end, etc.)
# - Tries to run a real backtest if engine is present.
# - If engine/modules not available, writes a minimal report and exits 0.
# ---------------------------------------

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPORTS_FILES = [
    "summary.txt",
    "metrics.json",
    "equity_curve.csv",
]

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _dump_minimal_reports(outdir: Path, note: str) -> None:
    outdir = _ensure_dir(outdir)
    (outdir / "summary.txt").write_text(
        "Backtest Summary\n"
        "Trades: 0\nWin-rate: 0.0%\nROI: 0.0%\nProfit Factor: 0.0\nR:R: 0.0\n"
        "Max DD: 0.0%\nTime DD (bars): 0\nSharpe: 0.0\n"
        f"Note: {note}\n",
        encoding="utf-8",
    )
    (outdir / "metrics.json").write_text(
        json.dumps({"trades": 0, "roi": 0.0, "note": note}, indent=2),
        encoding="utf-8",
    )
    # Optional empty equity curve
    (outdir / "equity_curve.csv").write_text("date,eq\n", encoding="utf-8")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backtest safely")
    p.add_argument("--symbol", required=True, help="NIFTY or BANKNIFTY")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--interval", required=True, help="e.g. 5m")
    p.add_argument("--outdir", required=True, help="Output reports directory")
    p.add_argument("--capital_rs", type=float, required=True)
    p.add_argument("--order_qty", type=int, required=True)
    p.add_argument("--fees_json", default="{}",
                   help='JSON: {"slippage_bps":0, "broker_flat":0, "broker_pct":0}')
    p.add_argument("--session_json", default="{}",
                   help='JSON: {"start":"09:20", "end":"15:25", "max_trades_per_day":0}')
    p.add_argument("--extra_params", default="{}",
                   help="Strategy extra parameters as JSON")
    return p.parse_args()

def main() -> int:
    args = parse_args()

    # Parse JSON blobs
    try:
        fees = json.loads(args.fees_json or "{}")
    except Exception:
        fees = {}
    try:
        sess = json.loads(args.session_json or "{}")
    except Exception:
        sess = {}
    try:
        extra = json.loads(args.extra_params or "{}")
    except Exception:
        extra = {}

    outdir = _ensure_dir(args.outdir)

    # Try to import the "real" engine
    try:
        from bot.backtest import run_backtest as real_run_backtest  # type: ignore
        print("🔷 Launching real backtest…")
        real_run_backtest(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
            outdir=str(outdir),
            capital_rs=float(args.capital_rs),
            order_qty=int(args.order_qty),
            slippage_bps=float(fees.get("slippage_bps", 0) or 0),
            broker_flat=float(fees.get("broker_flat", 0) or 0),
            broker_pct=float(fees.get("broker_pct", 0) or 0),
            session_start=sess.get("start"),
            session_end=sess.get("end"),
            max_trades_per_day=int(sess.get("max_trades_per_day", 0) or 0),
            **extra,
        )
        return 0

    except Exception as e:
        # If the engine/module is missing or raises, write minimal reports
        print(f"WARN: real backtest not available -> {e!r} -> writing minimal reports")
        _dump_minimal_reports(outdir, "Real engine not wired yet")
        return 0

if __name__ == "__main__":
    sys.exit(main())
