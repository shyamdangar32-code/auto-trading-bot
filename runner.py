# runner.py
import os
from datetime import datetime
import pandas as pd

from bot.config import get_cfg
from bot.utils import ensure_dir, save_json, load_json, send_telegram
from bot.data_io import prices
from bot.indicators import add_indicators
from bot.strategy import build_signals
from bot.backtest import backtest  # if you renamed to run_backtest, keep your import


CFG = get_cfg()
OUT = CFG.get("out_dir", "reports")
ensure_dir(OUT)

def status_line(last_row: pd.Series, label: str) -> str:
    dt = str(last_row.get("Date", ""))[:19]
    px = float(last_row["Close"])
    return f"📢 {CFG['symbol']} | {dt} | Signal: {label} | Price: {px:.2f}"

def _fmt_metrics(m: dict) -> str:
    # Guard against missing keys
    n  = m.get("n_trades", 0)
    wr = m.get("win_rate", 0.0)
    rr = m.get("rr", 0.0)
    roi = m.get("roi_pct", m.get("roi", 0.0))
    dd = m.get("max_dd_pct", 0.0)
    sh = m.get("sharpe", None)
    parts = [
        f"Trades: {n}",
        f"Win%: {wr:.1f}",
        f"R:R: {rr:.2f}",
        f"ROI%: {roi:.2f}",
        f"MaxDD%: {dd:.2f}",
    ]
    if isinstance(sh, (int, float)):
        parts.append(f"Sharpe: {sh:.2f}")
    return " | ".join(parts)

def main():
    df = prices(
        symbol=CFG["symbol"],
        period=CFG["lookback"],
        interval=CFG["interval"],
        zerodha_enabled=bool(CFG.get("zerodha_enabled", False)),
        zerodha_instrument_token=CFG.get("zerodha_instrument_token"),
    )

    df = add_indicators(df, CFG)
    df = build_signals(df, CFG)
    metrics = backtest(df, CFG)

    last = df.iloc[-1].copy()
    if "Date" in last.index:
        last["Date"] = str(last["Date"])

    latest_json_path = f"{OUT}/latest.json"
    prev = load_json(latest_json_path) or {}
    prev_label = str(prev.get("last_label", ""))

    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": CFG,
        "metrics": metrics,
        "last_label": str(last.get("label", "")),
        "last_row": last.to_dict(),
    }
    save_json(payload, latest_json_path)
    df.tail(250).to_csv(f"{OUT}/latest_signals.csv", index=False)

    label = str(last.get("label", ""))
    print("📊 Backtest:", metrics)
    print(status_line(last, label))

    # send alert only when signal changes and is not HOLD
    if label and (label != prev_label) and (label != "HOLD"):
        msg = (
            status_line(last, label)
            + "\n"
            + "🧮 " + _fmt_metrics(metrics)
        )
        send_telegram(msg)
    else:
        print("ℹ️ No alert sent (unchanged or HOLD).")

if __name__ == "__main__":
    main()
