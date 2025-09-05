from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bot.data_io import get_zerodha_ohlc
from bot.strategy import prepare_signals          # <-- ADD THIS
from bot.backtest import run_backtest, save_reports


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backtest via GitHub Actions")
    p.add_argument("--underlying", required=True)
    p.add_argument("--start", required=True)     # YYYY-MM-DD
    p.add_argument("--end", required=True)       # YYYY-MM-DD
    p.add_argument("--interval", default="1m")   # 1m/3m/5m/15m/day/...
    p.add_argument("--capital_rs", type=float, default=100000.0)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_block", default="backtest_loose")
    return p.parse_args()


def _ensure_dirs(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    return p


def main() -> None:
    args = _parse_args()
    outdir = _ensure_dirs(args.outdir)

    print(f"🧾 Fetching Zerodha OHLC: symbol={args.underlying} interval={args.interval} {args.start} -> {args.end}")
    try:
        prices = get_zerodha_ohlc(args.underlying, args.start, args.end, args.interval)
    except Exception as e:
        raise RuntimeError(f"Failed fetching OHLC for {args.underlying}: {e}") from e

    if prices.empty:
        (outdir / "metrics.json").write_text(json.dumps({"error": "no_data"}), encoding="utf-8")
        raise SystemExit("No OHLC data returned — check Zerodha credentials, token, or date range.")

    cfg = {
        "capital_rs": float(args.capital_rs),
        "order_qty": int(args.order_qty),
        "paper_trading": True,
        "live_trading": False,
        "backtest": {},  # optional tuning goes here
    }

    # ---- Map block → profile ----
    profile = "loose"
    if args.use_block.endswith("medium"):
        profile = "medium"
    elif args.use_block.endswith("strict"):
        profile = "strict"

    # ---- Prepare signals ----
    df = prepare_signals(prices, cfg, profile=profile)

    print(f"⚙️  Trade config: {{'order_qty': {cfg['order_qty']}, 'capital_rs': {cfg['capital_rs']}}}")
    print(f"🧱 Using profile/block: {args.use_block} (signals profile='{profile}')")

    summary, trades_df, equity_ser = run_backtest(df, cfg, use_block=args.use_block)

    save_reports(outdir=outdir, summary=summary, trades_df=trades_df, equity_ser=equity_ser)

    print("✅ Backtest finished & reports saved.")


if __name__ == "__main__":
    main()
