import os, json, yaml
from datetime import datetime
import pandas as pd

from bot.utils import ensure_dir, save_json, load_json, send_telegram
from bot.data_io import yahoo_prices
from bot.indicators import add_indicators
from bot.strategy import build_signals
from bot.backtest import backtest

CFG = yaml.safe_load(open("config.yaml", "r"))
OUT = CFG.get("out_dir", "reports")
ensure_dir(OUT)

def status_line(last_row: pd.Series, label: str) -> str:
    dt  = str(last_row.get("Date", ""))[:19]
    px  = float(last_row["Close"])
    return f"📢 {CFG['symbol']} | {dt} | Signal: {label} | Price: {px:.2f}"

def main():
    # 1) data
    df = yahoo_prices(CFG["symbol"], CFG["lookback"], CFG["interval"])

    # 2) features & signals
    df = add_indicators(df, CFG)
    df = build_signals(df, CFG)
    metrics = backtest(df, CFG)

    # 3) outputs
    last = df.iloc[-1]
    label = str(last["label"])
    latest_json_path = f"{OUT}/latest.json"
    prev = load_json(latest_json_path) or {}
    prev_label = prev.get("last_label", None)

    # save files (committed by CI later)
    ensure_dir(OUT)
    save_json({
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": CFG,
        "metrics": metrics,
        "last_label": label,
        "last_row": df.iloc[-1].to_dict()
    }, latest_json_path)
    df.tail(250).to_csv(f"{OUT}/latest_signals.csv", index=False)

    print("📊 Backtest:", metrics)
    print(status_line(last, label))

    # 4) alert only when changed (and not HOLD)
    if label != prev_label and label != "HOLD":
        send_telegram(status_line(last, label) + f"\nPnL (sum): {metrics['total_PnL']} | Trades: {metrics['n_trades']}")
    else:
        print("ℹ️ No alert sent (unchanged or HOLD).")

if __name__ == "__main__":
    main()
