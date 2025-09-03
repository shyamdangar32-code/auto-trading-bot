# tools/run_backtest.py
from __future__ import annotations
import os, argparse, json
from datetime import datetime, date, timedelta
import pandas as pd
from kiteconnect import KiteConnect

from bot.backtest import run_backtest, save_reports

def ist_day_bounds(d: date):
    return datetime(d.year, d.month, d.day, 9, 15), datetime(d.year, d.month, d.day, 15, 30)

def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)

def map_interval(x: str) -> str:
    m = (x or "").lower()
    return {"1m":"minute","1minute":"minute","3m":"3minute","5m":"5minute","10m":"10minute","15m":"15minute"}.get(m,"5minute")

def fetch_range_ohlc(kite: KiteConnect, token: int, d0: date, d1: date, interval: str) -> pd.DataFrame:
    frames = []
    for d in daterange(d0, d1):
        s,e = ist_day_bounds(d)
        try:
            raw = kite.historical_data(token, s, e, interval=interval)
        except Exception as ex:
            print(f"⚠️  {d}: {ex}")
            continue
        if not raw: 
            continue
        df = pd.DataFrame(raw)
        df["datetime"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        frames.append(df[["datetime","Open","High","Low","Close","Volume"]])
    if not frames:
        return pd.DataFrame(columns=["datetime","Open","High","Low","Close","Volume"]).set_index("datetime")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return out

def load_cfg():
    try:
        from bot.config import load_config  # optional if you have helper
        return load_config()
    except Exception:
        import yaml
        with open("config.yaml","r") as f:
            return yaml.safe_load(f) or {}

def main():
    ap = argparse.ArgumentParser(description="Run index backtest via Zerodha OHLC")
    ap.add_argument("--underlying", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--capital_rs", type=float, default=100000)
    ap.add_argument("--order_qty", type=int, default=1)
    ap.add_argument("--outdir", default="./reports")
    ap.add_argument("--profile", default="loose", help="loose|medium|strict")
    # tolerant to unknown args from workflow
    args, _ = ap.parse_known_args()

    d0 = datetime.fromisoformat(args.start).date()
    d1 = datetime.fromisoformat(args.end).date()
    os.makedirs(args.outdir, exist_ok=True)

    cfg = load_cfg() or {}
    cfg["capital_rs"] = args.capital_rs
    cfg["order_qty"]  = args.order_qty

    api_key = (os.getenv("ZERODHA_API_KEY") or cfg.get("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or cfg.get("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise SystemExit("Missing ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)

    # Token fallback by underlying
    token = int((cfg.get("intraday_options") or {}).get("index_token") or (260105 if args.underlying.upper()=="BANKNIFTY" else 256265))
    kc_interval = map_interval(args.interval)

    print(f"📥 Fetching Zerodha OHLC: token={token} interval={kc_interval} {d0} -> {d1}")
    prices = fetch_range_ohlc(kite, token, d0, d1, kc_interval)
    if prices.empty:
        raise SystemExit("No candles fetched.")

    print("⚙️  Trade config:", {
        "order_qty": cfg.get("order_qty"),
        "capital_rs": cfg.get("capital_rs"),
    })
    print(f"🧱 Using profile/block: backtest_{args.profile}")

    summary, trades_df, equity_ser = run_backtest(prices, cfg, profile=args.profile)

    save_reports(args.outdir, summary, trades_df, equity_ser, meta={
        "underlying": args.underlying,
        "start": args.start, "end": args.end, "interval": args.interval,
        "profile": args.profile
    })

    with open(os.path.join(args.outdir,"latest.json"),"w",encoding="utf-8") as f:
        json.dump({"n_rows":int(prices.shape[0]),"n_trades":int(summary.get("n_trades",0))}, f, indent=2)

    print("✅ Backtest complete"); print("   Summary:", summary); print("   Files:", args.outdir)

if __name__ == "__main__":
    main()
