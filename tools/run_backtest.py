#!/usr/bin/env python3
"""
Backtest runner (safe defaults)

Why this file exists:
- GitHub 'workflow_dispatch' has a hard limit of 10 inputs.
- To keep the workflow form simple (<= 6 inputs), we absorb the rest of the
  parameters here with sensible defaults (and allow env overrides).

It tries to call your real backtest implementation if available.
If anything fails (e.g., data source not reachable), it writes a minimal
report so the workflow never crashes and Telegram still gets an update.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from datetime import datetime, time

# -------------------------
# Defaults (can be overridden via ENV)
# -------------------------
DEF_SLIPPAGE_BPS   = int(os.getenv("SLIPPAGE_BPS", "0"))        # e.g., 5 = 0.05%
DEF_BROKER_FLAT    = float(os.getenv("BROKER_FLAT", "0"))       # ₹ per order
DEF_BROKER_PCT     = float(os.getenv("BROKER_PCT", "0"))        # e.g., 0.02 = 0.02%
DEF_SESSION_START  = os.getenv("SESSION_START", "09:20")        # HH:MM local
DEF_SESSION_END    = os.getenv("SESSION_END", "15:25")          # HH:MM local
DEF_MAX_TRADES_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "0"))  # 0 = unlimited

# Safe title used by telegram_notify.py when present
REPORT_TITLE = "Backtest"

# -------------------------
# Utilities
# -------------------------
def parse_hhmm(s: str) -> time:
    try:
        hh, mm = [int(x) for x in s.split(":")]
        return time(hh, mm)
    except Exception:
        # fall back to whole session if malformed
        return None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Minimal report writers
# -------------------------
def write_minimal_reports(out_dir: Path, title: str, note: str, meta: dict):
    """
    Creates the minimum set of files our tooling expects so that
    downstream steps (artifacts + Telegram) never fail.
    """
    ensure_dir(out_dir)
    # summary.json
    summary = {
        "title": title,
        "trades": 0,
        "win_rate": 0.0,
        "roi": 0.0,
        "profit_factor": 0.0,
        "rr": 0.0,
        "max_dd_pct": 0.0,
        "time_dd_bars": 0,
        "sharpe": 0.0,
        "note": note,
        "meta": meta,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # human friendly text too
    (out_dir / "SUMMARY.txt").write_text(
        f"{title}\n"
        f"Trades: 0\n"
        f"Note: {note}\n"
    )


# -------------------------
# Attempt real backtest (if available in repo)
# -------------------------
def try_real_backtest(args, params, out_dir: Path) -> bool:
    """
    Tries to import your project's backtest entry point and run it.
    Return True if succeeded; False if we should fall back to minimal.
    """
    # Persist run parameters (useful for debugging)
    ensure_dir(out_dir)
    (out_dir / "run_args.json").write_text(json.dumps({
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "interval": args.interval,
        "capital_rs": args.capital_rs,
        "order_qty": args.order_qty,
        "params": params
    }, indent=2))

    # Try a few common entry points to stay compatible
    try:
        # Preferred: bot.backtest has a 'run_backtest' or 'run' we can call
        try:
            from bot import backtest as bt_mod
        except Exception:
            bt_mod = None

        callable_fn = None
        if bt_mod is not None:
            for fn_name in ("run_backtest", "run"):
                if hasattr(bt_mod, fn_name):
                    callable_fn = getattr(bt_mod, fn_name)
                    break

        if callable_fn is None:
            # Fallback to tools/backtest-like function if user has it
            try:
                import importlib
                tb = importlib.import_module("backtest")
                for fn_name in ("run_backtest", "run"):
                    if hasattr(tb, fn_name):
                        callable_fn = getattr(tb, fn_name)
                        break
            except Exception:
                callable_fn = None

        if callable_fn is None:
            print("WARN: No backtest entry point found; writing minimal reports.")
            return False

        # Call the function in a generic way. Most repos accept something like:
        # run(symbol=..., start=..., end=..., interval=..., capital_rs=..., order_qty=..., out_dir=..., **params)
        print("ℹ️  Launching real backtest…")
        callable_fn(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            interval=args.interval,
            capital_rs=float(args.capital_rs),
            order_qty=int(args.order_qty),
            out_dir=str(out_dir),
            **params
        )
        print("✅ Real backtest finished.")
        return True

    except Exception as e:
        print(f"ERROR: real backtest failed → {e!r}")
        return False


# -------------------------
# CLI
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run index options backtest (with safe defaults).")
    p.add_argument("--symbol", required=True, help="NIFTY or BANKNIFTY")
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end",   required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--interval", required=True, help="e.g., 5m, 15m")
    p.add_argument("--capital_rs", required=True, help="Starting capital in ₹")
    p.add_argument("--order_qty",  required=True, help="Order quantity (lots or units)")
    p.add_argument("--out_dir", default="./reports", help="Output directory for reports")
    return p


def main():
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(Path("logs"))

    # Expand defaults + env overrides here
    session_start = parse_hhmm(DEF_SESSION_START)
    session_end   = parse_hhmm(DEF_SESSION_END)

    params = dict(
        slippage_bps=DEF_SLIPPAGE_BPS,
        broker_flat=DEF_BROKER_FLAT,
        broker_pct=DEF_BROKER_PCT,
        session_start=DEF_SESSION_START if session_start else None,
        session_end=DEF_SESSION_END if session_end else None,
        max_trades_per_day=DEF_MAX_TRADES_DAY,
    )

    meta = {
        "used_defaults": params,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    # Try to run the real backtest; if not possible, write a minimal report
    ok = try_real_backtest(args, params, out_dir)
    if not ok:
        note = "No data or runner fallback. (Backtest entry not available or failed.)"
        write_minimal_reports(out_dir, REPORT_TITLE, note, meta)


if __name__ == "__main__":
    main()
