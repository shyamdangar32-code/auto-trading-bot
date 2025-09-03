# tools/run_backtest.py
# Backtest driver: fetches index-level OHLC from Zerodha and calls bot.backtest

from __future__ import annotations
import os, sys, argparse, json
from datetime import datetime, date, timedelta

import pandas as pd
from kiteconnect import KiteConnect

# our repo modules
try:
    from bot.backtest import run_backtest, save_reports
except Exception as e:
    print("❌ Import bot.backtest failed:", e)
    raise

def ist_day_bounds(d: date):
    # Zerodha candles are IST; naive datetimes okay on Actions
    start = datetime(d.year, d.month, d.day, 9, 15, 0)
    end   = datetime(d.year, d.month, d.day, 15, 30, 0)
    return start, end

def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def map_interval(arg: str) -> str:
    m = (arg or "").lower().strip()
    return {
        "1m": "minute",
        "3m": "3minute",
        "5m": "5minute",
        "10m": "10minute",
        "15m": "15minute",
        "1minute": "minute",
        "3minute": "3minute",
        "5minute": "5minute",
        "10minute": "10minute",
        "15minute": "15minute",
    }.get(m, "5minute")

def fetch_range_ohlc(kite: KiteConnect, token: int, d0: date, d1: date, interval: str) -> pd.DataFrame:
    """Loop day by day to avoid window limits; concat to one DataFrame."""
    frames: list[pd.DataFrame] = []
    for d in daterange(d0, d1):
        # skip weekends quickly
        if d.weekday() >= 5:  # 5=Sat,6=Sun
            continue
        start, end = ist_day_bounds(d)
        try:
            raw = kite.historical_data(token, start, end, interval=interval)
        except Exception as e:
            print(f"⚠️  Skip {d}: {e}")
            continue
        if not raw:
            continue
        df = pd.DataFrame(raw)
        # Normalize columns for backtester
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        frames.append(df[["datetime", "Open", "High", "Low", "Close", "Volume"]])
    if not frames:
        return pd.DataFrame(columns=["datetime","Open","High","Low","Close","Volume"]).set_index("datetime")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
    return out

def load_cfg():
    # single source of truth
    try:
        from bot.config import load_config  # type: ignore
        return load_config()
    except Exception:
        import yaml
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}

def main():
    parser = argparse.ArgumentParser(description="Run index-level backtest via Zerodha OHLC")

    # Primary inputs (kept compatible with your workflow)
    parser.add_argument("--underlying", required=True, help="NIFTY or BANKNIFTY")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--interval", default="5m", help="1m,3m,5m,10m,15m")
    parser.add_argument("--capital_rs", type=float, default=100000)
    parser.add_argument("--order_qty", type=int, default=1)
    parser.add_argument("--mode", default="offline")

    # --- CI placeholders / compat flags (ignored by engine but accepted so errors stop) ---
    parser.add_argument("--fees_json", default="{}", help="(CI placeholder) ignored")
    parser.add_argument("--session_json", default="{}", help="(CI placeholder) ignored")
    parser.add_argument("--extra_params", default="{}", help="(CI placeholder) ignored")

    # Both spellings supported (old --out_dir vs new --outdir)
    parser.add_argument("--outdir", default="./reports", dest="outdir", help="Output directory")
    parser.add_argument("--out_dir", dest="outdir")  # keep compatibility with older workflow

    # Optional risk knobs (if passed from workflow; safe defaults)
    parser.add_argument("--slippage_bps", type=float, default=0.0)
    parser.add_argument("--broker_flat", type=float, default=0.0)
    parser.add_argument("--broker_pct", type=float, default=0.0)
    parser.add_argument("--session_start", default="09:15")
    parser.add_argument("--session_end", default="15:30")
    parser.add_argument("--max_trades_per_day", type=int, default=9999)

    # IMPORTANT: be tolerant to unknown args from workflows
    args, _unknown = parser.parse_known_args()

    # Dates
    d0 = datetime.fromisoformat(args.start).date()
    d1 = datetime.fromisoformat(args.end).date()
    if d1 < d0:
        raise SystemExit("End date must be >= start date")

    os.makedirs(args.outdir, exist_ok=True)

    # Config
    cfg = load_cfg() or {}
    # ensure capital/qty override from inputs
    cfg["capital_rs"] = args.capital_rs
    cfg["order_qty"]  = args.order_qty

    # Zerodha auth (env-driven on Actions)
    api_key = (os.getenv("ZERODHA_API_KEY") or cfg.get("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or cfg.get("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise SystemExit("Missing ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN (GitHub Secrets).")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)

    # Index token (from config.intraday_options)
    idx_token = None
    try:
        idx_token = int((cfg.get("intraday_options") or {}).get("index_token") or 0)
    except Exception:
        idx_token = 0

    # Basic fallback if not present
    if not idx_token:
        u = (args.underlying or "").upper()
        if u == "BANKNIFTY":
            idx_token = 260105
        elif u == "NIFTY":
            idx_token = 256265
        else:
            raise SystemExit("No index_token in config.yaml and unknown underlying; please set intraday_options.index_token")

    kc_interval = map_interval(args.interval)
    print(f"📥 Fetching Zerodha OHLC: token={idx_token} interval={kc_interval} {d0} → {d1}")
    prices = fetch_range_ohlc(kite, idx_token, d0, d1, kc_interval)
    if prices.empty:
        raise SystemExit("No candles fetched for the chosen range.")

    # Echo minimal trade config for traceability
    print("⚙️  Trade config:", {
        "order_qty": cfg.get("order_qty"),
        "capital_rs": cfg.get("capital_rs"),
        "reentry_max": (cfg.get("backtest", {}).get("reentry", {}) or {}).get("max_per_day"),
        "cooldown_bars": (cfg.get("backtest", {}).get("reentry", {}) or {}).get("cooldown_bars"),
        "stop_atr_mult": (cfg.get("backtest", {}).get("exits", {}) or {}).get("stop_atr_mult"),
        "take_atr_mult": (cfg.get("backtest", {}).get("exits", {}) or {}).get("take_atr_mult"),
        "max_trades_per_day": (cfg.get("backtest", {}).get("guardrails", {}) or {}).get("max_trades_per_day"),
    })

    # Run backtest (your strategy/indicators are wired inside bot.strategy / bot.backtest)
    summary, trades_df, equity_ser = run_backtest(prices, cfg)

    # Persist reports
    save_reports(args.outdir, summary, trades_df, equity_ser)

    # Small metadata echo for CI logs
    try:
        meta = {
            "underlying": args.underlying,
            "start": args.start,
            "end": args.end,
            "interval": args.interval,
            "n_rows": int(prices.shape[0]),
            "n_trades": int(summary.get("n_trades", 0)) if isinstance(summary, dict) else None,
        }
        with open(os.path.join(args.outdir, "latest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print("ℹ️ Could not write latest.json (non-fatal):", e)

    print("✅ Backtest complete")
    print("   Summary:", summary)
    print(f"   Files written in: {args.outdir}")

if __name__ == "__main__":
    main()
