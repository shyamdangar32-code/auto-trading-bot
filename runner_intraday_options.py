# runner_intraday_options.py
from __future__ import annotations

import os, sys, json, math
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------
# make imports robust for different repo layouts
# (root, tools/, bot/, core_backtest/)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CANDIDATE_DIRS = [
    ROOT,
    ROOT / "tools",
    ROOT / "bot",
    ROOT / "core_backtest",
]

for d in CANDIDATE_DIRS:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

# data_io import (multiple fallbacks)
dio = None
_import_errs = []
for modpath in ("data_io", "tools.data_io", "bot.data_io", "core_backtest.data_io"):
    try:
        dio = __import__(modpath, fromlist=["*"])
        break
    except Exception as e:
        _import_errs.append(f"{modpath}: {e}")

if dio is None:
    print("WARN: data_io import failed; fallbacks tried ->")
    for line in _import_errs:
        print("  -", line)
    # continue; runner will create minimal reports so workflow doesn’t break

# lightweight TA (RSI)
def rsi(series, period=14):
    import pandas as pd
    delta = series.diff()
    up = (delta.clip(lower=0)).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=os.getenv("INDEX_SYMBOL", "BANKNIFTY"))
    ap.add_argument("--start", required=False, help="YYYY-MM-DD")
    ap.add_argument("--end", required=False, help="YYYY-MM-DD")
    ap.add_argument("--interval", default=os.getenv("INTERVAL", "5minute"))
    ap.add_argument("--mode", default=os.getenv("SIGNAL_MODE", "balanced"))
    ap.add_argument("--outdir", default="reports")
    return ap.parse_args()

def utc_date(s: str) -> datetime:
    # safe parse to aware datetime (tzinfo=UTC)
    dt = datetime.strptime(s, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)

def fetch_index_df(symbol: str, start: datetime, end: datetime, interval: str):
    """Use data_io if available, else return None."""
    if dio is None:
        return None
    # try common function names
    for fn in ("get_index_candles", "get_index_ohlc", "load_index_candles"):
        f = getattr(dio, fn, None)
        if callable(f):
            try:
                df = f(symbol=symbol, start=start, end=end, interval=interval)
                return df
            except TypeError:
                # some impls use positional args
                try:
                    df = f(symbol, start, end, interval)
                    return df
                except Exception:
                    pass
            except Exception:
                pass
    return None

def backtest_index_rsi(df, mode="balanced"):
    """
    very simple intraday-like index strategy:
    - entry: RSI crosses above 55 -> long; crosses below 45 -> short
    - exit: opposite cross OR EOD
    - single position at a time
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return pd.DataFrame(), {
            "n_trades": 0, "win_rate": 0.0, "roi_pct": 0.0,
            "profit_factor": 0.0, "rr": 0.0, "max_dd_pct": 0.0,
            "time_dd_bars": 0, "sharpe_ratio": 0.0,
            "note": "No data; runner fallback."
        }

    df = df.copy()
    # expect columns: date/time index + 'close'
    price = df["close"].astype(float)
    df["rsi"] = rsi(price, 14)
    df["side"] = 0
    df.loc[(df["rsi"] > 55) & (df["rsi"].shift(1) <= 55), "side"] = 1
    df.loc[(df["rsi"] < 45) & (df["rsi"].shift(1) >= 45), "side"] = -1

    # build trades
    trades = []
    pos = 0
    entry_px = None
    entry_ts = None
    for ts, row in df.iterrows():
        side = int(row["side"])
        px = float(row["close"])
        if pos == 0:
            if side != 0:
                pos = side
                entry_px = px
                entry_ts = ts
        else:
            # flip or EOD exit
            flip = (pos == 1 and side == -1) or (pos == -1 and side == 1)
            is_eod = False
            if isinstance(ts, datetime):
                is_eod = ts.time() >= datetime.strptime("15:25", "%H:%M").time()
            if flip or is_eod:
                pnl = (px - entry_px) * pos
                trades.append({"timestamp": ts, "action": "exit", "price": px, "pnl": pnl})
                pos = 0
                entry_px = None
                entry_ts = None
            # if flip, immediately open new
            if flip:
                pos = side
                entry_px = px
                entry_ts = ts

    import pandas as pd
    sigs = pd.DataFrame(trades)

    # metrics (very compact)
    if len(sigs) == 0:
        return sigs, {
            "n_trades": 0, "win_rate": 0.0, "roi_pct": 0.0,
            "profit_factor": 0.0, "rr": 0.0, "max_dd_pct": 0.0,
            "time_dd_bars": 0, "sharpe_ratio": 0.0,
            "note": "No signals generated today."
        }

    wins = (sigs["pnl"] > 0).sum()
    losses = (sigs["pnl"] < 0).sum()
    gross_profit = sigs.loc[sigs["pnl"] > 0, "pnl"].sum()
    gross_loss = -sigs.loc[sigs["pnl"] < 0, "pnl"].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = (wins / len(sigs)) * 100.0

    # simple ROI assuming 1 unit per trade on index
    capital = 100000.0
    roi_pct = (sigs["pnl"].sum() / capital) * 100.0

    # rudimentary rr estimate
    avg_win = sigs.loc[sigs["pnl"] > 0, "pnl"].mean() if wins else 0.0
    avg_loss = -sigs.loc[sigs["pnl"] < 0, "pnl"].mean() if losses else 1e-9
    rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    # pseudo drawdown
    eq = sigs["pnl"].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = dd.min() if len(dd) else 0.0
    max_dd_pct = (max_dd / capital) * 100.0
    time_dd_bars = int((dd < 0).sum())

    # naive Sharpe (no RF)
    import numpy as np
    ret = sigs["pnl"].replace(0, 1e-9) / capital
    sharpe = (ret.mean() / (ret.std() + 1e-9)) * math.sqrt(252*78/5)  # ~5min bars/day scaling

    metrics = {
        "n_trades": int(len(sigs)),
        "win_rate": float(round(win_rate, 2)),
        "roi_pct": float(round(roi_pct, 2)),
        "profit_factor": float(round(profit_factor, 2)) if profit_factor != float("inf") else 0.0,
        "rr": float(round(rr, 2)),
        "max_dd_pct": float(round(max_dd_pct, 2)),
        "time_dd_bars": int(time_dd_bars),
        "sharpe_ratio": float(round(sharpe, 2)),
        "note": "",
    }
    return sigs, metrics

def save_reports(outdir: Path, signals_df, metrics):
    import pandas as pd
    outdir.mkdir(parents=True, exist_ok=True)
    # latest_signals.csv
    sigp = outdir / "latest_signals.csv"
    if isinstance(signals_df, pd.DataFrame) and len(signals_df):
        signals_df.to_csv(sigp, index=False)
    else:
        sigp.write_text("timestamp,action,price,pnl\n", encoding="utf-8")
    # metrics.json
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # latest.json (tiny run summary for bot)
    latest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "summary": metrics.get("note", "") or "OK",
        "config": {},
    }
    (outdir / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")

def main():
    args = parse_args()

    # date range defaults: last 1 month (UTC) if not provided
    if args.start:
        start = utc_date(args.start)
    else:
        start = (datetime.now(timezone.utc) - timedelta(days=30)).replace(hour=9, minute=15, second=0, microsecond=0)
    if args.end:
        end = utc_date(args.end) + timedelta(days=1)  # inclusive end day
    else:
        end = datetime.now(timezone.utc)

    df = fetch_index_df(args.symbol, start, end, args.interval)

    if df is None or len(df) == 0:
        print("WARN: data_io.get_index_candles unavailable or returned empty → returning minimal reports.")
        signals_df = None
        metrics = {
            "n_trades": 0, "win_rate": 0.0, "roi_pct": 0.0,
            "profit_factor": 0.0, "rr": 0.0, "max_dd_pct": 0.0,
            "time_dd_bars": 0, "sharpe_ratio": 0.0,
            "note": "No data; runner fallback."
        }
        save_reports(Path(args.outdir), signals_df, metrics)
        print("✅ minimal reports written.")
        return

    signals_df, metrics = backtest_index_rsi(df, mode=args.mode)
    save_reports(Path(args.outdir), signals_df, metrics)
    print("✅ reports written to", Path(args.outdir).resolve())

if __name__ == "__main__":
    main()
