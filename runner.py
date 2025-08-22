# runner.py
import os
from datetime import datetime

import pandas as pd

from bot.config import get_cfg
from bot.utils import ensure_dir, save_json, load_json, send_telegram
from bot.data_io import prices          # Zerodha-first, Yahoo fallback
from bot.indicators import add_indicators
from bot.strategy import build_signals
from bot.backtest import backtest


# ---------- config & output setup ----------

CFG = get_cfg()                          # reads config.yaml + env overrides
OUT = CFG.get("out_dir", "reports")
ensure_dir(OUT)


# ---------- small pretty line ----------

def status_line(last_row: pd.Series, label: str) -> str:
    dt = str(last_row.get("Date", ""))[:19]   # stringify to avoid Timestamp issues
    px = float(last_row["Close"])
    return f"📢 {CFG['symbol']} | {dt} | Signal: {label} | Price: {px:.2f}"


# ---------- main workflow ----------

def main():
    # 1) Load prices (Zerodha if enabled, else Yahoo). If both fail, notify & exit gracefully.
    try:
        df = prices(
            symbol=CFG["symbol"],
            period=CFG["lookback"],
            interval=CFG["interval"],
            zerodha_enabled=bool(CFG.get("zerodha_enabled", False)),
            zerodha_instrument_token=CFG.get("zerodha_instrument_token"),
        )
    except Exception as e:
        msg = f"❗ Data fetch failed: {e}. No trading today. If using Zerodha, refresh ACCESS_TOKEN."
        print(msg)
        send_telegram(msg)
        ensure_dir(OUT)
        save_json({"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                   "error": str(e),
                   "config": CFG}, f"{OUT}/latest.json")
        return

    # 2) Features & signals
    df = add_indicators(df, CFG)
    df = build_signals(df, CFG)
    metrics = backtest(df, CFG)

    # 3) Outputs
    last = df.iloc[-1].copy()
    if "Date" in last.index:
        last["Date"] = str(last["Date"])      # JSON-safe

    latest_json_path = f"{OUT}/latest.json"
    prev = load_json(latest_json_path) or {}
    prev_label = prev.get("last_label")

    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": CFG,
        "metrics": metrics,
        "last_label": str(last.get("label", "")),
        "last_row": last.to_dict(),
    }
    save_json(payload, latest_json_path)
    df.tail(250).to_csv(f"{OUT}/latest_signals.csv", index=False)

    # Logs
    label = str(last.get("label", ""))
    print("📊 Backtest:", metrics)
    print(status_line(last, label))

    # 4) Alert only when signal changes and is not HOLD
    if label != prev_label and label != "HOLD":
        msg = status_line(last, label) + f"\nPnL (sum): {metrics['total_PnL']} | Trades: {metrics['n_trades']}"
        send_telegram(msg)
    else:
        print("ℹ️ No alert sent (unchanged or HOLD).")


if __name__ == "__main__":
    main()
