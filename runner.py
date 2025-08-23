# runner.py
import os
from datetime import datetime, timezone
import pandas as pd
import pytz

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

# default label if missing
if not str(CFG.get("symbol", "")).strip():
    CFG["symbol"] = "^NSEI"

IST = pytz.timezone(CFG.get("tz", "Asia/Kolkata"))


# --------------- pretty printer ---------------
def status_line(last_row: pd.Series, label: str) -> str:
    dt = str(last_row.get("Date", ""))[:19]
    px = float(last_row["Close"])
    return f"📢 {CFG['symbol']} | {dt} | Signal: {label} | Price: {px:.2f}"


# --------------- optional “yesterday-only” cut ---------------
def apply_paper_cutoff(df: pd.DataFrame) -> pd.DataFrame:
    """
    If simulate_yesterday_only=true, drop today's rows (in IST) and keep <= yesterday.
    Or if paper_trade_cutoff=YYYY-MM-DD set, keep rows <= that date (IST).
    """
    cut_cfg = str(CFG.get("paper_trade_cutoff", "")).strip()
    y_only = bool(CFG.get("simulate_yesterday_only", False))

    if not (cut_cfg or y_only):
        return df

    if cut_cfg:
        cutoff_date = IST.localize(datetime.strptime(cut_cfg, "%Y-%m-%d")).date()
    else:
        now_ist = datetime.now(IST)
        cutoff_date = (now_ist.date() - pd.Timedelta(days=1)).date()

    df2 = df.copy()
    # ensure Date is timezone-aware/naive consistently
    if not pd.api.types.is_datetime64_any_dtype(df2["Date"]):
        df2["Date"] = pd.to_datetime(df2["Date"])
    # compare by date in IST
    df2_ist_date = df2["Date"].dt.tz_convert(IST).dt.date if df2["Date"].dt.tz is not None else df2["Date"].dt.tz_localize(IST).dt.date
    mask = df2_ist_date <= cutoff_date
    cut_df = df2.loc[mask].copy()
    print(f"✂️ Paper cutoff <= {cutoff_date} IST → kept {len(cut_df)}/{len(df2)} rows")
    return cut_df if not cut_df.empty else df


# --------------- main workflow ---------------
def main():
    # 1) Data load (Zerodha only)
    df = prices(
        symbol=CFG["symbol"],
        period=CFG["lookback"],
        interval=CFG["interval"],
        zerodha_enabled=bool(CFG.get("zerodha_enabled", True)),
        zerodha_instrument_token=CFG.get("zerodha_instrument_token"),
    )

    # optional: keep only up-to cutoff (paper backtest for 1 day / yesterday)
    df = apply_paper_cutoff(df)

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
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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

    # send alert only when signal changes and is not HOLD
    if label and (label != prev_label) and (label != "HOLD"):
        msg = status_line(last, label) + f"\nPnL (sum): {metrics['total_PnL']} | Trades: {metrics['n_trades']}"
        send_telegram(msg)
    else:
        print("ℹ️ No alert sent (unchanged or HOLD).")


if __name__ == "__main__":
    main()
