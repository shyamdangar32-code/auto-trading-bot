#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Telegram summary for backtest reports.
- Reads metrics.json and normalizes key names (ROI vs roi_pct, R_R vs rr, max_dd_perc vs max_dd_pct, sharpe vs sharpe_ratio).
- If critical fields are missing/zero, recompute from trades.csv and augment.
- Sends summary + all PNG/JPG images in the reports dir.
"""

import os
import sys
import json
import glob
import time
import argparse
import requests
import pandas as pd
import numpy as np

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))

# ---------- utils ----------

def eprint(*a): print(*a, file=sys.stderr)

def read_textfile(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def _metrics_from_trades(trades_csv, initial_capital=INITIAL_CAPITAL):
    """Strong fallback: compute metrics from trades.csv only."""
    try:
        if not os.path.isfile(trades_csv):
            return {}
        df = pd.read_csv(trades_csv)
        if df.empty:
            return {}

        # robust PnL column detection
        candidates = ["pnl", "netpnl", "net_pnl", "pnl_rs", "profit", "pl", "p&l"]
        pnl_col = next((c for c in df.columns if any(k in c.lower() for k in candidates)), None)
        if pnl_col is None:
            eprint("WARN: No PnL-like column found in trades.csv")
            return {}

        # closed/filled trades if status present
        if "status" in df.columns:
            closed = df[df["status"].astype(str).str.lower().isin(
                ["closed", "filled", "exit", "complete"]
            )].copy()
            if closed.empty:
                closed = df.copy()
        else:
            closed = df.copy()

        trades = int(len(closed))
        wins = int((closed[pnl_col] > 0).sum())
        losses = int((closed[pnl_col] < 0).sum())
        winrate = 100 * wins / trades if trades else 0.0

        equity = initial_capital + closed[pnl_col].cumsum()
        final_capital = float(equity.iloc[-1]) if trades else float(initial_capital)
        roi_pct = 100 * (final_capital - initial_capital) / initial_capital

        peak = equity.cummax()
        dd = (equity - peak) / peak
        maxdd_pct = 100 * float(dd.min()) if trades else 0.0
        time_dd_bars = int((dd < 0).sum()) if trades else 0

        rets = closed[pnl_col] / float(initial_capital)
        sharpe = float((rets.mean() / (rets.std() or 1.0)) * np.sqrt(len(rets))) if trades else 0.0

        total_profit = float(closed.loc[closed[pnl_col] > 0, pnl_col].sum())
        total_loss = float(-closed.loc[closed[pnl_col] < 0, pnl_col].sum())
        profit_factor = (total_profit / total_loss) if total_loss > 0 else (float("inf") if total_profit > 0 else 0.0)

        avg_win = float(closed.loc[closed[pnl_col] > 0, pnl_col].mean()) if wins else 0.0
        avg_loss = float(closed.loc[closed[pnl_col] < 0, pnl_col].abs().mean()) if losses else 0.0
        rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        return {
            "trades": trades,
            "win_rate": round(winrate, 2),
            "roi_pct": round(roi_pct, 2),
            "final_capital": round(final_capital, 2),
            "profit_factor": (round(profit_factor, 2) if np.isfinite(profit_factor) else "Inf"),
            "rr": round(rr, 2),
            "max_dd_pct": round(maxdd_pct, 2),
            "time_dd_bars": time_dd_bars,
            "sharpe_ratio": round(sharpe, 2),
        }
    except Exception as ex:
        eprint("WARN: _metrics_from_trades failed:", ex)
        return {}

def _normalize_metrics_keys(m: dict) -> dict:
    """Map alternate keys to canonical names used in the message."""
    norm = {}
    norm["trades"]        = int(m.get("trades", m.get("n_trades", 0) or 0))
    norm["win_rate"]      = float(m.get("win_rate", m.get("winrate", 0) or 0))
    norm["roi_pct"]       = float(m.get("roi_pct", m.get("ROI", 0) or 0))
    norm["profit_factor"] = m.get("profit_factor", 0)
    norm["rr"]            = float(m.get("rr", m.get("R_R", 0) or 0))
    norm["max_dd_pct"]    = float(m.get("max_dd_pct", m.get("max_dd_perc", 0) or 0))
    norm["time_dd_bars"]  = int(m.get("time_dd_bars", m.get("time_dd", 0) or 0))
    norm["sharpe_ratio"]  = float(m.get("sharpe_ratio", m.get("sharpe", 0) or 0))
    norm["final_capital"] = float(m.get("final_capital", m.get("FinalCapital", 0) or 0))
    return norm

# ---------- build message ----------

def build_summary(rep_dir: str) -> str:
    metrics_path = os.path.join(rep_dir, "metrics.json")

    # 0) metrics.json first
    if os.path.isfile(metrics_path):
        try:
            m = json.load(open(metrics_path, "r", encoding="utf-8"))
            norm = _normalize_metrics_keys(m)

            # augment from trades.csv if critical numbers look zero/missing
            need_aug = (
                norm["trades"] == 0
                or norm["roi_pct"] == 0
                or norm["max_dd_pct"] == 0
                or norm["sharpe_ratio"] == 0
                or norm["rr"] == 0
            )
            if need_aug:
                tcsv = os.path.join(rep_dir, "trades.csv")
                m2 = _metrics_from_trades(tcsv)
                if m2:
                    for k in ["trades","win_rate","roi_pct","final_capital","profit_factor","rr","max_dd_pct","time_dd_bars","sharpe_ratio"]:
                        norm[k] = m2.get(k, norm.get(k, 0))

            return (
                "📊 <b>Backtest Summary</b>\n"
                f"• Trades: <b>{norm['trades']}</b>\n"
                f"• Win-rate: <b>{norm['win_rate']:.2f}%</b>\n"
                f"• ROI: <b>{norm['roi_pct']:.2f}%</b>\n"
                f"• Profit Factor: <b>{norm['profit_factor']}</b>\n"
                f"• R:R: <b>{norm['rr']:.2f}</b>\n"
                f"• Max DD: <b>{norm['max_dd_pct']:.2f}%</b>\n"
                f"• Time DD (bars): <b>{norm['time_dd_bars']}</b>\n"
                f"• Sharpe: <b>{norm['sharpe_ratio']:.2f}</b>"
            )
        except Exception as ex:
            eprint("WARN: metrics.json parse failed:", ex)

    # 1) trades.csv fallback (if no metrics.json)
    tcsv = os.path.join(rep_dir, "trades.csv")
    if os.path.isfile(tcsv):
        m = _metrics_from_trades(tcsv)
        if m:
            return (
                "📊 <b>Backtest Summary</b>\n"
                f"• Trades: <b>{m['trades']}</b>\n"
                f"• Win-rate: <b>{m['win_rate']:.2f}%</b>\n"
                f"• ROI: <b>{m['roi_pct']:.2f}%</b>\n"
                f"• Profit Factor: <b>{m['profit_factor']}</b>\n"
                f"• R:R: <b>{m['rr']:.2f}</b>\n"
                f"• Max DD: <b>{m['max_dd_pct']:.2f}%</b>\n"
                f"• Time DD (bars): <b>{m['time_dd_bars']}</b>\n"
                f"• Sharpe: <b>{m['sharpe_ratio']:.2f}</b>"
            )

    # 2) legacy fallbacks
    js_path = os.path.join(rep_dir, "summary.json")
    if os.path.isfile(js_path):
        try:
            data = json.load(open(js_path, "r", encoding="utf-8"))
            norm = _normalize_metrics_keys(data)
            return (
                "📊 <b>Backtest Summary</b>\n"
                f"• Trades: <b>{norm['trades']}</b>\n"
                f"• Win-rate: <b>{norm['win_rate']:.2f}%</b>\n"
                f"• ROI: <b>{norm['roi_pct']:.2f}%</b>\n"
                f"• Profit Factor: <b>{norm['profit_factor']}</b>\n"
                f"• R:R: <b>{norm['rr']:.2f}</b>\n"
                f"• Max DD: <b>{norm['max_dd_pct']:.2f}%</b>\n"
                f"• Time DD (bars): <b>{norm['time_dd_bars']}</b>\n"
                f"• Sharpe: <b>{norm['sharpe_ratio']:.2f}</b>"
            )
        except Exception as ex:
            eprint("WARN: summary.json parse failed:", ex)

    txt_path = os.path.join(rep_dir, "summary.txt")
    if os.path.isfile(txt_path):
        txt = read_textfile(txt_path)
        if txt:
            return f"📊 <b>Backtest Summary</b>\n{txt}"

    return "📊 <b>Backtest Summary</b>\nNote: No metrics found."

# ---------- telegram ----------

def tg_send(token, method, payload=None, files=None):
    url = f"https://api.telegram.org/bot{token}/{method}"
    r = requests.post(url, data=payload or {}, files=files, timeout=30)
    if r.status_code != 200:
        eprint("❌ Telegram HTTP", r.status_code, r.text)
        sys.exit(1)
    jr = r.json()
    if not jr.get("ok", False):
        eprint("❌ Telegram API error:", jr)
        sys.exit(1)
    print("✅ Telegram:", method, "OK")
    return jr

def send_message(token, chat_id, text):
    return tg_send(token, "sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    })

def send_photo(token, chat_id, path, caption=None):
    with open(path, "rb") as f:
        files = {"photo": (os.path.basename(path), f)}
        payload = {"chat_id": chat_id}
        if caption:
            payload["caption"] = caption
    return tg_send(token, "sendPhoto", payload=payload, files=files)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="path to reports dir")
    ap.add_argument("--token", required=True)
    ap.add_argument("--chat", required=True)
    ap.add_argument("--title", default="Backtest")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", required=True)
    args = ap.parse_args()

    rep_dir = args.reports
    title = f"{args.title} • {args.symbol} • {args.interval}"

    msg = build_summary(rep_dir)
    msg = f"<b>{title}</b>\n\n{msg}"

    # send message
    send_message(args.token, args.chat, msg)

    # send images
    pics = sorted(glob.glob(os.path.join(rep_dir, "*.png")))
    if not pics:
        pics = sorted(glob.glob(os.path.join(rep_dir, "*.jpg")))
    for p in pics:
        try:
            send_photo(args.token, args.chat, p, caption=os.path.basename(p))
            time.sleep(0.3)
        except Exception as ex:
            eprint("WARN: sending photo failed for", p, ex)

    print("✅ Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
