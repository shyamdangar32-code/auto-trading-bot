# tools/run_backtest.py
# Runner compatible with .github/workflows/backtest.yml
from __future__ import annotations

import os
import json
import argparse
from datetime import datetime, timedelta, date

import pandas as pd
from typing import Dict, Any

# project imports
from bot.backtest import run_backtest, save_reports  # noqa

# ----------------------------
# Zerodha / instrument mapping
# ----------------------------
INDEX_TOKENS = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
}

INTERVAL_MAP = {
    "1m": "minute",
    "3m": "3minute",
    "5m": "5minute",
    "10m": "10minute",
    "15m": "15minute",
    "30m": "30minute",
    "60m": "60minute",
    "1d": "day",
    "day": "day",
}

def ist_day_bounds(d: date) -> tuple[datetime, datetime]:
    # NOTE: Kite historicals want **naive** datetimes; no tzinfo.
    start = datetime(d.year, d.month, d.day, 9, 15, 0)
    end = datetime(d.year, d.month, d.day, 15, 30, 0)
    return start, end

def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def load_cfg_dict() -> Dict[str, Any]:
    """
    Load your trading config.
    Priority:
      1) bot.config.load_config() if present
      2) config.yaml if exists
      3) empty dict
    """
    # Try central loader first
    try:
        from bot.config import load_config  # type: ignore
        return dict(load_config() or {})
    except Exception:
        pass

    # Try YAML fallback
    if os.path.exists("config.yaml"):
        try:
            import yaml  # type: ignore
            with open("config.yaml", "r") as f:
                return dict(yaml.safe_load(f) or {})
        except Exception:
            pass

    return {}

def merge_cfg(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (overlay or {}).items():
        # ignore None/empty strings so defaults stay intact
        if v is not None and v != "":
            out[k] = v
    return out

def fetch_range_ohlc_online(underlying: str, d0: date, d1: date, interval: str) -> pd.DataFrame:
    """Fetch OHLC via Kite (requires ZERODHA_API_KEY & ZERODHA_ACCESS_TOKEN)."""
    api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    access = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise SystemExit("❌ Missing ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN envs.")

    # Import here so the script still imports in offline environments
    from kiteconnect import KiteConnect  # type: ignore

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)
    try:
        prof = kite.profile()
        print(f"✅ Zerodha token OK for user: {prof.get('user_name', 'unknown')}")
    except Exception as e:
        raise SystemExit(f"❌ Zerodha profile check failed: {e}")

    token = INDEX_TOKENS.get(underlying.upper())
    if token is None:
        raise SystemExit(f"❌ Unsupported underlying: {underlying}. Choose from {list(INDEX_TOKENS)}")

    iv = INTERVAL_MAP.get(interval.lower(), "5minute")

    frames: list[pd.DataFrame] = []
    for d in daterange(d0, d1):
        start, end = ist_day_bounds(d)
        try:
            raw = kite.historical_data(token, start, end, iv)
        except Exception as e:
            print(f"⚠️  Skip {d}: {e}")
            continue
        if not raw:
            continue
        df = pd.DataFrame(raw)
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        frames.append(df[["datetime", "Open", "High", "Low", "Close", "Volume"]])

    if not frames:
        print("⚠️ No candles fetched for the chosen range.")
        return pd.DataFrame(columns=["datetime", "Open", "High", "Low", "Close", "Volume"]).set_index("datetime")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("datetime").drop_duplicates("datetime")
    return out.set_index("datetime")

def main():
    ap = argparse.ArgumentParser(description="Backtest runner (workflow-compatible)")
    ap.add_argument("--underlying", required=True, help="NIFTY or BANKNIFTY")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--interval", required=True, help="1m/3m/5m/10m/15m/30m/60m/1d")
    ap.add_argument("--capital_rs", type=float, default=None)
    ap.add_argument("--order_qty", type=int, default=None)
    ap.add_argument("--mode", default="offline", choices=["offline", "online"])
    ap.add_argument("--extra_params", default="{}",
                    help='JSON string with extra params (e.g. {"leg_sl_percent":25,"telegram_notify":true})')
    ap.add_argument("--out_dir", default="./reports")
    args = ap.parse_args()

    # Parse dates
    d0 = datetime.strptime(args.start, "%Y-%m-%d").date()
    d1 = datetime.strptime(args.end, "%Y-%m-%d").date()

    # Load and merge configuration
    cfg = load_cfg_dict()
    override = {
        "underlying": args.underlying.upper(),
        "capital_rs": args.capital_rs,
        "order_qty": args.order_qty,
        "interval": args.interval.lower(),
        "mode": args.mode,
    }
    try:
        extra = json.loads(args.extra_params or "{}")
    except Exception as e:
        raise SystemExit(f"❌ Failed to parse --extra_params JSON: {e}")

    cfg = merge_cfg(cfg, override)
    cfg = merge_cfg(cfg, extra)

    print("⚙️  Effective config (selected keys):", {
        k: cfg.get(k) for k in [
            "underlying", "interval", "capital_rs", "order_qty",
            "leg_sl_percent", "combined_target_percent", "reentry_max",
            "reentry_cooldown", "trailing_enabled", "trail_type",
            "trail_atr_mult", "adx_min", "telegram_notify"
        ] if k in cfg
    })

    # Get price data
    if args.mode == "online":
        print(f"📥 Fetching {args.interval} OHLC for {args.underlying} {d0} → {d1} via Kite…")
        prices = fetch_range_ohlc_online(args.underlying, d0, d1, args.interval)
    else:
        # OFFLINE path – try project data loader; fall back with a clear message
        prices = None
        try:
            from bot.data_io import load_prices  # your project-defined loader (if exists)
            prices = load_prices(args.underlying, d0, d1, args.interval)  # type: ignore
        except Exception:
            pass
        if prices is None:
            raise SystemExit(
                "❌ Offline mode selected but no local loader found.\n"
                "   Implement bot.data_io.load_prices(underlying, start_date, end_date, interval)\n"
                "   or run with --mode online."
            )

    if prices is None or len(prices) == 0:
        raise SystemExit("❌ No price data available for the selected period.")

    print(f"✅ Price bars: {len(prices)}")

    # Run backtest
    print("▶ Running backtest…")
    summary, trades_df, equity_ser = run_backtest(prices, cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    save_reports(args.out_dir, summary, trades_df, equity_ser)

    print("📦 Reports written to:", args.out_dir)
    print("✅ Backtest complete.")
    print("   Summary:", summary)

if __name__ == "__main__":
    main()
