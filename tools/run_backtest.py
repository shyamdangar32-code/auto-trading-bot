# tools/run_backtest.py
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

import pandas as pd

# engine hooks (unchanged)
from bot.backtest import run_backtest, save_reports  # noqa: F401


# -------- Zerodha OHLC fetch (inline, no extra file) --------
def get_zerodha_ohlc(
    instrument_token: int,
    interval: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Fetch OHLC using KiteConnect directly.
    Expects ZERODHA_API_KEY & ZERODHA_ACCESS_TOKEN in env.
    """
    from kiteconnect import KiteConnect  # imported here to keep global deps light

    api_key = os.environ.get("ZERODHA_API_KEY", "").strip()
    access_token = os.environ.get("ZERODHA_ACCESS_TOKEN", "").strip()

    if not api_key or not access_token:
        raise RuntimeError(
            "Missing Zerodha creds: set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN in Actions secrets."
        )

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Kite intervals: "minute","3minute","5minute","15minute","60minute","day"
    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=start,
        to_date=end,
        interval=interval,
        continuous=False,
        oi=True,
    )

    if not data:
        raise RuntimeError("Empty OHLC returned from Zerodha.")

    df = pd.DataFrame(data)
    # 'date' field is ISO string; make TZ-naive index (our engine handles tz)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    ).set_index("Date").sort_index()

    # Ensure expected columns exist
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df:
            raise RuntimeError(f"Missing column '{col}' in Zerodha OHLC.")

    return df


# -------- Small helpers --------
INDEX_TOKENS: Dict[str, int] = {
    # Keep what you really use; BANKNIFTY was 260105 in your runs
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
    "FINNIFTY": 2707457,
}

INTERVAL_MAP = {
    "1m": "minute",
    "3m": "3minute",
    "5m": "5minute",
    "15m": "15minute",
    "60m": "60minute",
    "1d": "day",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run backtest")
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--interval", default="1m")
    p.add_argument("--capital_rs", type=float, default=100000)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--outdir", default="reports")
    p.add_argument("--use_block", default="backtest_loose")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    und = args.underlying.upper().strip()
    token = INDEX_TOKENS.get(und)
    if not token:
        raise SystemExit(f"Unsupported underlying '{und}'. Add its instrument token in INDEX_TOKENS.")

    interval_kite = INTERVAL_MAP.get(args.interval, args.interval)

    # Paths
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Minimal config passed to engine
    cfg = {
        "tz": "Asia/Kolkata",
        "out_dir": str(outdir),
        "capital_rs": float(args.capital_rs),
        "order_qty": int(args.order_qty),
        # backtest block mirrors (engine reads these)
        "backtest": {
            "session_start": "09:20",
            "session_end": "15:20",
            "slippage_bps": 2.0,
            "brokerage_flat": 20.0,
            "brokerage_pct": 0.0003,
            "ema_fast": 21,
            "ema_slow": 50,
            "rsi_len": 14,
            "rsi_buy": 52,
            "rsi_sell": 48,
            "atr_len": 14,
            "ema_poke_pct": 0.0001,
            "filters": {
                "adx_len": 14,
                "adx_min": 12,
                "use_htf": True,
                "htf_rule": "15min",
                "htf_ema_len": 20,
                "min_atr_pct": 0.0003,
                "session_start": "09:25",
                "session_end": "15:15",
            },
            "reentry": {
                "max_per_day": 20,
                "cooldown_bars": 0,
            },
            "guardrails": {
                "max_trades_per_day": 30,
                "max_daily_loss_rs": 2500,
                "stop_after_target_rs": 4000,
            },
            "exits": {
                "stop_atr_mult": 1.0,
                "take_atr_mult": 1.3,
                "trail": {"type": "atr", "atr_mult": 1.0, "step_bars": 3},
            },
        },
    }

    print(
        f"🗓  Fetching Zerodha OHLC: token={token} interval={interval_kite} "
        f"{args.start} -> {args.end}"
    )
    prices = get_zerodha_ohlc(token, interval_kite, args.start, args.end)

    print(
        f"⚙️  Trade config: {{'order_qty': {cfg['order_qty']}, 'capital_rs': {cfg['capital_rs']}}}"
    )
    print(f"📦 Using profile/block: {args.use_block}")

    # Run engine
    summary, trades_df, equity_ser = run_backtest(prices, cfg, use_block=args.use_block)

    # Persist artifacts
    save_reports(
        out_dir=outdir,
        prices=prices,
        equity=e

quity_ser,
        trades=trades_df,
        summary=summary,
        meta={
            "underlying": und,
            "interval": args.interval,
            "start": args.start,
            "end": args.end,
            "use_block": args.use_block,
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    )


if __name__ == "__main__":
    main()
