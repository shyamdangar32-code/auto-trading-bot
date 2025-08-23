# runner.py
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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

if not str(CFG.get("symbol", "")).strip():
    CFG["symbol"] = "NIFTYBEES.NS"


# --------------- helpers ---------------
def status_line(last_row: pd.Series, label: str) -> str:
    dt = str(last_row.get("Date", ""))[:19]
    px = float(last_row["Close"])
    return f"📢 {CFG['symbol']} | {dt} | Signal: {label} | Price: {px:.2f}"

def apply_cutoff_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limit data to 'yesterday' (India time) or to an explicit cutoff date
    so we can simulate yesterday’s market in paper mode.
    """
    tz_name = CFG.get("tz", "Asia/Kolkata")

    # explicit date takes priority if provided
    cutoff_cfg = str(CFG.get("paper_trade_cutoff", "")).strip()
    if cutoff_cfg:
        cutoff_date = pd.to_datetime(cutoff_cfg).date()
    elif bool(CFG.get("simulate_yesterday_only", False)):
        # compute yesterday in local exchange time
        now_local = datetime.now(ZoneInfo(tz_name))
        cutoff_date = (now_local.date() - timedelta(days=1))
    else:
        return df  # no cutoff requested

    # ensure Date is datetime, then filter by date <= cutoff_date
    dts = pd.to_datetime(df["Date"])
    df = df[dts.dt.date <= cutoff_date].copy()
    if df.empty:
        raise RuntimeError(f"No rows on or before cutoff {cutoff_date}.")
    print(f"🧪 Paper-trade cutoff active -> using data up to {cutoff_date} ({tz_name}) "
          f"→ {len(df)} rows kept.")
    return df


# --------------- main workflow ---------------
def main():
    # 1) Data (Zerodha first; your token must be valid)
    df = prices(
        symbol=CFG["symbol"],
        period=CFG["lookback"],
        interval=CFG["interval"],
        zerodha_enabled=bool(CFG.get("zerodha_enabled", False)),
        zerodha_instrument_token=CFG.get("zerodha_instrument_token"),
    )

    # 1b) apply “yesterday only” (or explicit cutoff date)
    df = apply_cutoff_if_needed(df)

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

    # In “yesterday” simulation we still keep the normal rule:
    # only alert when the label changed and isn’t HOLD.
    if label and (label != prev_label) and (label != "HOLD"):
        msg = status_line(last, label) + \
              f"\nPnL (sum): {metrics['total_PnL']} | Trades: {metrics['n_trades']} | Win rate: {metrics['win_rate']:.1f}%"
        send_telegram(msg)
    else:
        print("ℹ️ No alert sent (unchanged or HOLD).")


if __name__ == "__main__":
    main()
