# tools/run_backtest.py
from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Tuple
import pandas as pd

# repo imports (engine + signals)
from bot.backtest import run_backtest, save_reports
from bot.strategy import prepare_signals

# Try to use your existing fetcher if present. (We removed the old bot.pricefeed import.)
def _fetch_zerodha(underlying: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Returns OHLC dataframe indexed by timezone-aware pandas DatetimeIndex.
    Must contain columns: Open, High, Low, Close, Volume.
    """
    # Prefer bot.data_io.get_zerodha_ohlc if available
    try:
        from bot.data_io import get_zerodha_ohlc  # your existing helper
        return get_zerodha_ohlc(
            symbol=underlying, start=start, end=end, interval=interval,
            api_key=os.environ.get("ZERODHA_API_KEY",""),
            api_secret=os.environ.get("ZERODHA_API_SECRET",""),
            access_token=os.environ.get("ZERODHA_ACCESS_TOKEN",""),
        )
    except Exception as e:
        raise RuntimeError(f"Failed fetching OHLC for {underlying}: {e}") from e


def _load_cfg() -> dict:
    """
    Loads config.yaml if present; otherwise returns a safe default dict.
    """
    import yaml
    cfg_path = pathlib.Path("config.yaml")
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    # sensible defaults (mirrors you shared)
    return {
        "tz": "Asia/Kolkata",
        "capital_rs": 100000,
        "order_qty": 1,
        "ema_fast": 21, "ema_slow": 50,
        "rsi_len": 14, "rsi_buy": 52, "rsi_sell": 48,
        "atr_len": 14, "ema_poke_pct": 0.0001,
        "backtest": {
            "session_start": "09:20", "session_end": "15:20",
            "slippage_bps": 2.0, "brokerage_flat": 20.0, "brokerage_pct": 0.0003,
            "ema_fast": 21, "ema_slow": 50, "rsi_len": 14, "rsi_buy": 52, "rsi_sell": 48,
            "atr_len": 14, "ema_poke_pct": 0.0001,
            "filters": {
                "adx_len": 14, "adx_min": 15,
                "use_htf": True, "htf_rule": "15min", "htf_ema_len": 20,
                "min_atr_pct": 0.0003,
                "session_start": "09:25", "session_end": "15:15",
            },
            "reentry": {"max_per_day": 20, "cooldown_bars": 0},
            "guardrails": {"max_trades_per_day": 30, "max_daily_loss_rs": 2500, "stop_after_target_rs": 4000},
            "exits": {"stop_atr_mult": 1.0, "take_atr_mult": 1.3, "trail": {"type":"atr","atr_mult":1.0,"step_bars":3}},
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--underlying", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="1m")
    p.add_argument("--capital_rs", type=float, default=100000)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_block", default="backtest_loose")  # e.g. backtest_loose|backtest_medium|backtest_strict
    args = p.parse_args()

    # Fetch prices
    print(f"🗂 Fetching Zerodha OHLC: symbol={args.underlying} interval={args.interval} {args.start} -> {args.end}")
    prices = _fetch_zerodha(args.underlying, args.start, args.end, args.interval)
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise RuntimeError("Empty price dataframe returned from fetcher")
    prices = prices.sort_index()

    # Load config
    cfg = _load_cfg()
    cfg["capital_rs"] = float(args.capital_rs)
    cfg["order_qty"] = int(args.order_qty)

    # Determine profile from use_block suffix
    prof = "loose"
    if args.use_block.endswith("strict"): prof = "strict"
    elif args.use_block.endswith("medium"): prof = "medium"

    # Build signals
    df = prepare_signals(prices, cfg, profile=prof)

    # Run engine
    summary, trades_df, equity_ser = run_backtest(df, cfg, use_block=args.use_block)

    # Save outputs
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_reports(outdir, summary, trades_df, equity_ser)

    print("✅ Backtest finished.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
