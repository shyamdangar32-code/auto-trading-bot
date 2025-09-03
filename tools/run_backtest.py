# tools/run_backtest.py
# Backtest driver: fetches index-level OHLC from Zerodha and calls bot.backtest

from __future__ import annotations
import os, argparse, json
from datetime import datetime, date, timedelta
import pandas as pd
from kiteconnect import KiteConnect

from bot.backtest import run_backtest, save_reports

def ist_day_bounds(d: date):
    return datetime(d.year, d.month, d.day, 9, 15, 0), datetime(d.year, d.month, d.day, 15, 30, 0)

def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def map_interval(arg: str) -> str:
    m = (arg or "").lower().strip()
    return {
        "1m": "minute", "3m": "3minute", "5m": "5minute",
        "10m": "10minute", "15m": "15minute",
        "1minute":"minute","3minute":"3minute","5minute":"5minute",
        "10minute":"10minute","15minute":"15minute"
    }.get(m, "5minute")

def fetch_range_ohlc(kite: KiteConnect, token: int, d0: date, d1: date, interval: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in daterange(d0, d1):
        if d.weekday() >= 5:  # Sat/Sun
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
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        frames.append(df[["datetime","Open","High","Low","Close","Volume"]])
    if not frames:
        return pd.DataFrame(columns=["datetime","Open","High","Low","Close","Volume"]).set_index("datetime")
    out = pd.concat(frames, ignore_index=True).sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
    return out

def load_cfg():
    try:
        from bot.config import load_config  # if you have helper
        return load_config()
    except Exception:
        import yaml
        with open("config.yaml","r") as f:
            return yaml.safe_load(f) or {}

def main():
    p = argparse.ArgumentParser(description="Run index-level backtest via Zerodha OHLC")
    p.add_argument("--underlying", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="5m")
    p.add_argument("--capital_rs", type=float, default=100000)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--mode", default="offline")
    # NEW: choose profile/block
    p.add_argument("--use_block", default="backtest_medium", help="backtest_loose | backtest_medium | backtest_strict")

    # ignore-any extra CI flags
    p.add_argument("--outdir", default="./reports", dest="outdir")
    p.add_argument("--out_dir", dest="outdir")
    p.add_argument("--slippage_bps", type=float, default=0.0)
    p.add_argument("--broker_flat", type=float, default=0.0)
    p.add_argument("--broker_pct", type=float, default=0.0)
    p.add_argument("--session_start", default="09:15")
    p.add_argument("--session_end", default="15:30")
    p.add_argument("--max_trades_per_day", type=int, default=9999)
    args, _ = p.parse_known_args()

    d0 = datetime.fromisoformat(args.start).date()
    d1 = datetime.fromisoformat(args.end).date()
    if d1 < d0:
        raise SystemExit("End date must be >= start date")
    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_cfg() or {}
    cfg["capital_rs"] = args.capital_rs
    cfg["order_qty"]  = args.order_qty

    api_key = (os.getenv("ZERODHA_API_KEY") or cfg.get("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or cfg.get("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise SystemExit("Missing ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN (GitHub Secrets).")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)

    u = (args.underlying or "").upper()
    idx_token = int((cfg.get("intraday_options") or {}).get("index_token") or (260105 if u=="BANKNIFTY" else 256265))
    kc_interval = map_interval(args.interval)

    print(f"📥 Fetching Zerodha OHLC: token={idx_token} interval={kc_interval} {d0} → {d1}")
    prices = fetch_range_ohlc(kite, idx_token, d0, d1, kc_interval)
    if prices.empty:
        raise SystemExit("No candles fetched for the chosen range.")

    print("⚙️  Trade config:", {"order_qty":cfg.get("order_qty"), "capital_rs":cfg.get("capital_rs")})
    print(f"👟 Using profile/block: {args.use_block}")

    summary, trades_df, equity_ser = run_backtest(prices, cfg, use_block=args.use_block)
    save_reports(args.outdir, summary, trades_df, equity_ser)

    try:
        meta = {
            "underlying": args.underlying, "start": args.start, "end": args.end,
            "interval": args.interval, "use_block": args.use_block,
            "n_rows": int(prices.shape[0]),
            "n_trades": int(summary.get("n_trades", 0)) if isinstance(summary, dict) else None,
        }
        with open(os.path.join(args.outdir, "latest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print("ℹ️ Could not write latest.json (non-fatal):", e)

    print("✅ Backtest complete")
    print("   Summary:", summary)
    print(f"   Files in: {args.outdir}")

if __name__ == "__main__":
    main()
