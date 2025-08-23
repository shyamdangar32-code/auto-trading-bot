# runner.py
from datetime import datetime
import pandas as pd

from bot.config import get_cfg
from bot.utils import ensure_dir, save_json, load_json, send_telegram
from bot.data_io import prices
from bot.indicators import add_indicators
from bot.strategy import build_signals
from bot.backtest import backtest


# --------------- config & output ---------------
CFG = get_cfg()                          # loads config.yaml + env overrides
OUT = CFG.get("out_dir", "reports")
ensure_dir(OUT)

# --------------- pretty printer ---------------
def status_line(last_row: pd.Series, label: str) -> str:
    dt = str(last_row.get("Date", ""))[:19]
    px = float(last_row["Close"])
    return f"📢 {CFG['symbol']} | {dt} | Signal: {label} | Price: {px:.2f}"


# --------------- main workflow ---------------
def main():
    try:
        # 1) Data (Zerodha only)
        df = prices(
            symbol=CFG["symbol"],
            period=CFG["lookback"],
            interval=CFG["interval"],
            zerodha_enabled=bool(CFG.get("zerodha_enabled", False)),
            zerodha_instrument_token=CFG.get("zerodha_instrument_token"),
        )
    except Exception as e:
        # Token invalid / env missing / any fetch issue -> notify & exit gracefully
        msg = f"❗ Data fetch failed: {e}\n(If using Zerodha, refresh ACCESS_TOKEN.)"
        print(msg)
        send_telegram(msg)
        # write a tiny heartbeat so artifacts exist
        save_json({"timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                   "error": str(e)}, f"{OUT}/latest.json")
        return

    # 2) Features & signals
    df = add_indicators(df, CFG)
    df = build_signals(df, CFG)
    metrics = backtest(df, CFG)

    # 3) Save outputs
    last = df.iloc[-1].copy()
    if "Date" in last.index:
        last["Date"] = str(last["Date"])  # JSON-safe

    latest_json_path = f"{OUT}/latest.json"
    prev_payload = load_json(latest_json_path) or {}
    prev_label = str(prev_payload.get("last_label", ""))

    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": CFG,
        "metrics": metrics,
        "last_label": str(last.get("label", "")),
        "last_row": last.to_dict(),
    }
    save_json(payload, latest_json_path)
    df.tail(250).to_csv(f"{OUT}/latest_signals.csv", index=False)

    # 4) Logs + optional Telegram alert
    label = str(last.get("label", ""))
    print("📊 Backtest:", metrics)
    print(status_line(last, label))

    if label and (label != prev_label) and (label != "HOLD"):
        msg = (f"📅 {payload['timestamp']}\n" +
               status_line(last, label) +
               f"\nPnL(sum): {metrics['total_PnL']} | Trades: {metrics['n_trades']} | Win rate: {metrics['win_rate']:.1f}%")
        send_telegram(msg)
    else:
        print("ℹ️ No alert sent (unchanged or HOLD).")


if __name__ == "__main__":
    main()
