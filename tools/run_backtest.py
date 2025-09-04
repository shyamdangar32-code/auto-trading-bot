# tools/run_backtest.py
# Backtest driver: fetches index-level OHLC from Zerodha and calls bot.backtest

from __future__ import annotations
import os, sys, argparse, json, time, random
from datetime import datetime, date, timedelta

import pandas as pd
from kiteconnect import KiteConnect

# our repo modules
try:
    from bot.backtest import run_backtest, save_reports
except Exception as e:
    print("❌ Import bot.backtest failed:", e)
    raise

# ---------------- Utils ----------------

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

# ------------- Rate-limit aware fetch -------------

def fetch_range_ohlc(kite: KiteConnect, token: int, d0: date, d1: date, interval: str) -> pd.DataFrame:
    """
    Pull OHLC day-by-day so Zerodha window limits are respected.
    Retries with exponential backoff on 429/Too many requests and other transient errors.
    """
    frames: list[pd.DataFrame] = []
    for d in daterange(d0, d1):
        if d.weekday() >= 5:  # 5=Sat,6=Sun
            continue

        start, end = ist_day_bounds(d)
        tries, backoff = 0, 2.0
        while True:
            try:
                raw = kite.historical_data(token, start, end, interval=interval)
                # tiny jitter between calls to be nice to API
                time.sleep(0.30 + random.random() * 0.15)
                break
            except Exception as e:
                msg = str(e).lower()
                tries += 1
                # Zerodha sometimes throws plain text "Too many requests"
                if "too many request" in msg or "429" in msg:
                    wait = min(60.0, backoff)
                    print(f"⚠️  Rate limited on {d} — sleeping {wait:.1f}s then retry…")
                    time.sleep(wait)
                    backoff *= 1.8
                    if tries <= 6:
                        continue
                # transient network hiccups: retry few times
                if tries <= 3 and any(x in msg for x in ["timed out", "temporar", "network"]):
                    time.sleep(2.0 * tries)
                    continue
                print(f"⚠️  Skip {d}: {e}")
                raw = []
                break

        if not raw:
            continue

        df = pd.DataFrame(raw)
        df["datetime"] = pd.to_datetime(df["date"], utc=False)
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

# ---------------- Config ----------------

def load_cfg():
    # single source of truth
    try:
        from bot.config import load_config  # type: ignore
        return load_config()
    except Exception:
        import yaml
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Run index-level backtest via Zerodha OHLC")

    # Primary inputs
    parser.add_argument("--underlying", required=True, help="NIFTY or BANKNIFTY")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--interval", default="5m", help="1m,3m,5m,10m,15m")
    parser.add_argument("--capital_rs", type=float, default=100000)
    parser.add_argument("--order_qty", type=int, default=1)
    parser.add_argument("--outdir", default="./reports")
    # NEW: profile (loose | medium | strict)
    parser.add_argument("--profile", default="loose")

    # placeholders accepted (ignored)
    parser.add_argument("--fees_json", default="{}")
    parser.add_argument("--session_json", default="{}")
    parser.add_argument("--extra_params", default="{}")

    # Optional knobs
    parser.add_argument("--slippage_bps", type=float, default=0.0)
    parser.add_argument("--broker_flat", type=float, default=0.0)
    parser.add_argument("--broker_pct", type=float, default=0.0)
    parser.add_argument("--session_start", default="09:15")
    parser.add_argument("--session_end", default="15:30")
    parser.add_argument("--max_trades_per_day", type=int, default=9999)

    args, _unknown = parser.parse_known_args()

    # Dates
    d0 = datetime.fromisoformat(args.start).date()
    d1 = datetime.fromisoformat(args.end).date()
    if d1 < d0:
        raise SystemExit("End date must be >= start date")

    os.makedirs(args.outdir, exist_ok=True)

    # Config
    cfg = load_cfg() or {}
    cfg["capital_rs"] = args.capital_rs
    cfg["order_qty"]  = args.order_qty

    # Zerodha auth
    api_key = (os.getenv("ZERODHA_API_KEY") or cfg.get("ZERODHA_API_KEY") or "").strip()
    access  = (os.getenv("ZERODHA_ACCESS_TOKEN") or cfg.get("ZERODHA_ACCESS_TOKEN") or "").strip()
    if not api_key or not access:
        raise SystemExit("Missing ZERODHA_API_KEY / ZERODHA_ACCESS_TOKEN (GitHub Secrets).")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access)

    # Index token (from config.intraday_options or fallback)
    idx_token = 0
    try:
        idx_token = int((cfg.get("intraday_options") or {}).get("index_token") or 0)
    except Exception:
        idx_token = 0
    if not idx_token:
        u = (args.underlying or "").upper()
        idx_token = 260105 if u == "BANKNIFTY" else 256265 if u == "NIFTY" else 0
    if not idx_token:
        raise SystemExit("No index_token available; set intraday_options.index_token")

    kc_interval = map_interval(args.interval)
    print(f"📥 Fetching Zerodha OHLC: token={idx_token} interval={kc_interval} {d0} → {d1}")
    prices = fetch_range_ohlc(kite, idx_token, d0, d1, kc_interval)
    if prices.empty:
        raise SystemExit("No candles fetched for the chosen range.")

    print("🧱 Using profile/block:", f"backtest_{args.profile}")

    # Backtest (returns summary dict, trades df, equity series)
    summary, trades_df, equity_ser = run_backtest(prices, cfg, profile=args.profile)

    # Persist reports
    save_reports(args.outdir, summary, trades_df, equity_ser, profile=args.profile)

    # Metadata echo
    try:
        meta = {
            "underlying": args.underlying,
            "start": args.start,
            "end": args.end,
            "interval": args.interval,
            "profile": args.profile,
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
