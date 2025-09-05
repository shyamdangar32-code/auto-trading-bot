# tools/run_backtest.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

# Repo modules
from bot.data_io import get_zerodha_ohlc
from bot.backtest import run_backtest, save_reports  # uses your existing engine + writer


# ---------------------------- CLI & IO helpers --------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backtest via GitHub Actions")
    p.add_argument("--underlying", required=True, help="e.g. NIFTY or BANKNIFTY")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--interval", default="1m", help="1m/3m/5m/15m/day/...")
    p.add_argument("--capital_rs", type=float, default=100000.0)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--outdir", required=True, help="Output directory for reports")
    p.add_argument("--use_block", default="backtest_loose", help="Config/profile block to use")
    return p.parse_args()


def _fetch_zerodha(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    # 🔧 IMPORTANT: no extra api_key/api_secret/access_token args here.
    return get_zerodha_ohlc(symbol, start, end, interval)


def _ensure_dirs(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    return p


# ------------------------------- Main runner ----------------------------------

def main() -> None:
    args = _parse_args()
    outdir = _ensure_dirs(args.outdir)

    print(f"🧾 Fetching Zerodha OHLC: symbol={args.underlying} interval={args.interval} {args.start} -> {args.end}")

    try:
        prices = _fetch_zerodha(args.underlying, args.start, args.end, args.interval)
    except Exception as e:
        # Make the failure clear in logs for Actions summary
        raise RuntimeError(f"Failed fetching OHLC for {args.underlying}: {e}") from e

    if prices.empty:
        # Create a tiny stub so later steps don't crash, but exit non-zero to surface the issue
        (outdir / "metrics.json").write_text(json.dumps({"error": "no_data"}), encoding="utf-8")
        raise SystemExit("No OHLC data returned — check Zerodha credentials, token, or date range.")

    # Build a small runtime config overlay for the engine
    cfg = {
        "capital_rs": float(args.capital_rs),
        "order_qty": int(args.order_qty),
        "send_only_signals": True,
        "paper_trading": True,
        "live_trading": False,

        # Backtest block name (profile) is passed separately to engine
    }

    print(f"⚙️  Trade config: {{'order_qty': {cfg['order_qty']}, 'capital_rs': {cfg['capital_rs']}}}")
    print(f"🧱 Using profile/block: {args.use_block}")

    # Your engine returns: summary (dict), trades_df (DataFrame), equity_ser (Series)
    summary, trades_df, equity_ser = run_backtest(
        prices,
        cfg,
        use_block=args.use_block,
    )

    # Persist reports (charts, CSVs, metrics.json, markdown, etc.)
    save_reports(
        outdir=outdir,
        prices=prices,
        summary=summary,
        trades_df=trades_df,
        equity_ser=equity_ser,
        profile_name=args.use_block,
        underlying=args.underlying,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )

    print("✅ Backtest finished & reports saved.")


if __name__ == "__main__":
    main()
