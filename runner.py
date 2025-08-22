# runner.py
import os
from datetime import datetime
import pandas as pd

from bot.config import get_cfg
from bot.utils import ensure_dir, save_json, load_json, send_telegram
from bot.data_io import prices
from bot.indicators import add_indicators
from bot.strategy import build_signals
from bot.backtest import backtest

CFG = get_cfg()
OUT = CFG.get("out_dir", "reports")
ensure_dir(OUT)

# --- Runtime mode ---
ENV_LIVE = os.getenv("LIVE_MODE", "").strip() == "1"
ENV_FORCE_PAPER = os.getenv("FORCE_PAPER", "1").strip() == "1"  # default: paper

LIVE_MODE = (CFG.get("live_trading", False) or ENV_LIVE) and not ENV_FORCE_PAPER
PAPER_MODE = not LIVE_MODE

def status_line(last_row: pd.Series, label: str) -> str:
    dt = str(last_row.get("Date", ""))[:19]
    px = float(last_row["Close"])
    mode = "LIVE" if LIVE_MODE else "PAPER"
    return f"📢 Zerodha | {dt} | {mode} | Signal: {label} | Price: {px:.2f}"

def append_paper_trade(row: pd.Series, label: str, path: str):
    cols = ["Date","Open","High","Low","Close","ema_f","ema_s","rsi","adx","atr","label"]
    data = {k: row[k] for k in row.index if k in cols}
    data["Date"] = str(row.get("Date",""))
    data["signal_label"] = label
    data["ts_logged_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    import csv
    header_needed = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data.keys()))
        if header_needed:
            w.writeheader()
        w.writerow(data)

def maybe_send_telegram(msg: str):
    try:
        send_telegram(msg)
    except Exception as e:
        print("⚠️ Telegram send failed:", repr(e))

def main():
    # 1) Zerodha prices only
    df = prices(
        symbol=None,   # not used now
        period=CFG["lookback"],
        interval=CFG["interval"],
        zerodha_enabled=True,
        zerodha_instrument_token=CFG.get("zerodha_instrument_token"),
    )

    # 2) Indicators + signals
    df = add_indicators(df, CFG)
    df = build_signals(df, CFG)
    metrics = backtest(df, CFG)

    # 3) Save snapshot
    last = df.iloc[-1].copy()
    if "Date" in last.index:
        last["Date"] = str(last["Date"])
    prev = load_json(f"{OUT}/latest.json") or {}
    prev_label = str(prev.get("last_label", ""))

    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {**CFG, "effective_mode": "LIVE" if LIVE_MODE else "PAPER"},
        "metrics": metrics,
        "last_label": str(last.get("label", "")),
        "last_row": last.to_dict(),
    }
    save_json(payload, f"{OUT}/latest.json")
    df.tail(250).to_csv(f"{OUT}/latest_signals.csv", index=False)

    # 4) Logs
    label = str(last.get("label", ""))
    print("📊 Backtest:", metrics)
    line = status_line(last, label)
    print(line)

    if PAPER_MODE and label and label != "HOLD":
        append_paper_trade(df.iloc[-1], label, os.path.join(OUT, "paper_trades.csv"))

    if label and (label != prev_label) and (label != "HOLD"):
        msg = line + f"\nPnL (sum): {metrics['total_PnL']} | Trades: {metrics['n_trades']}"
        maybe_send_telegram(msg)
    else:
        print("ℹ️ No alert sent (unchanged or HOLD).")

if __name__ == "__main__":
    main()
