# runner.py  — Zerodha-only + clear "yesterday" alert if requested

import os
from datetime import datetime
import pandas as pd

from bot.config import get_cfg
from bot.utils import ensure_dir, save_json, load_json, send_telegram
from bot.data_io import zerodha_prices         # <-- Zerodha only
from bot.indicators import add_indicators
from bot.strategy import build_signals
from bot.backtest import backtest


# --------------- config & output ---------------
CFG = get_cfg()                          # loads config.yaml + env overrides
OUT = CFG.get("out_dir", "reports")
ensure_dir(OUT)

# safety defaults
if not str(CFG.get("symbol", "")).strip():
    CFG["symbol"] = "NIFTYBEES.NS"

TZ = CFG.get("tz", "Asia/Kolkata")


# --------------- helpers ---------------
def _status_line(last_row: pd.Series, label: str) -> str:
    dt = str(last_row.get("Date", ""))[:19]
    px = float(last_row["Close"])
    return f"📢 {CFG['symbol']} | {dt} | Signal: {label} | Price: {px:.2f}"


def _apply_paper_cutoff(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    If simulate_yesterday_only or paper_trade_cutoff is set, trim the dataframe
    to candles <= that local date (end of day). Returns (df_trimmed, note).
    """
    cutoff_note = None
    sim_yday = bool(CFG.get("simulate_yesterday_only", False))
    cutoff_str = CFG.get("paper_trade_cutoff", "")

    if not sim_yday and not cutoff_str:
        return df, cutoff_note

    if cutoff_str:
        # Specific date
        cutoff_local = pd.Timestamp(f"{cutoff_str} 23:59:59", tz=TZ)
        cutoff_note = f"🧪 Paper-trade cutoff -> using data up to {cutoff_str} ({TZ})"
    else:
        # Yesterday (local)
        now_local = pd.Timestamp.now(tz=TZ)
        yday = (now_local - pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        cutoff_local = yday
        cutoff_note = f"🧪 Paper-trade cutoff -> using data up to {yday.date()} ({TZ})"

    # Make df['Date'] timezone-aware in local tz for a clean compare
    dates = df["Date"]
    if dates.dt.tz is None:
        dates_local = dates.dt.tz_localize(TZ)
    else:
        dates_local = dates.dt.tz_convert(TZ)

    kept = df[dates_local <= cutoff_local].copy()
    print(f"{cutoff_note} -> {len(kept)} rows kept.")

    return kept, cutoff_note


# --------------- main workflow ---------------
def main():
    # ---- guard rails for Zerodha-only run
    token = CFG.get("zerodha_instrument_token")
    if not token:
        raise RuntimeError("Missing zerodha_instrument_token in config.yaml")

    # 1) Data (Zerodha only)
    df = zerodha_prices(int(token), CFG["lookback"], CFG["interval"])

    # Optional: trim to yesterday / specific date for paper test
    df, cutoff_note = _apply_paper_cutoff(df)

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

    # 4) Logs
    label = str(last.get("label", ""))
    print("📊 Backtest:", metrics)
    if cutoff_note:
        print(cutoff_note)
    print(_status_line(last, label))

    # 5) Telegram alert
    #    - Always send if a cutoff is active (to show the yesterday result clearly)
    #    - Otherwise: only when signal changes and not HOLD (old behaviour)
    always_send = bool(cutoff_note)
    changed = (label and (label != prev_label) and (label != "HOLD"))

    if always_send or changed:
        header = "🧪 Yesterday (paper backtest)" if cutoff_note else "🔔 Signal update"
        wr = metrics.get("win_rate")
        wr_txt = f"{wr:.1f}%" if isinstance(wr, (int, float)) else str(wr)
        msg = (
            f"{header}\n"
            f"{_status_line(last, label)}\n"
            f"PnL(sum): {metrics['total_PnL']} | Trades: {metrics['n_trades']} | Win rate: {wr_txt}"
        )
        send_telegram(msg)
    else:
        print("ℹ️ No alert sent (unchanged or HOLD).")


if __name__ == "__main__":
    main()
