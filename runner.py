# runner.py
import os
import json
from datetime import datetime

import pandas as pd

from bot.config import get_cfg
from bot.utils import ensure_dir, save_json, load_json, send_telegram
from bot.data_io import prices          # unified (Zerodha-first, Yahoo fallback)
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
    # 1) Load prices (Zerodha if enabled, else Yahoo)
    df = prices(
        symbol=CFG["symbol"],
        period=CFG["lookback"],
        interval=CFG["interval"],
        zerodha_enabled=bool(CFG.get("zerodha_enabled", False)),
        zerodha_instrument_token=CFG.get("zerodha_instrument_token"),
    )

    # 2) Features & signals
    df = add_indicators(df, CFG)
    df = build_signals(df, CFG)
    metrics = backtest(df, CFG)

    # 3) Outputs to disk (and remember previous label for change detection)
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

    # ---------- messaging ----------
    label = str(last.get("label", ""))
    is_change = (label != prev_label)
    ts_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # One compact summary for every run
    summary = (
        f"📅 {ts_utc}\n"
        f"{status_line(last, label)}\n"
        f"PnL(sum): {metrics['total_PnL']:.2f} | "
        f"Trades: {metrics['n_trades']} | "
        f"Win rate: {metrics['win_rate']:.1f}%"
    )

    # If there’s a new actionable signal, shout it at the top
    if is_change and label != "HOLD":
        summary = f"🚨 NEW SIGNAL: {label}\n" + summary

    print(summary)
    send_telegram(summary)


if __name__ == "__main__":
    main()
