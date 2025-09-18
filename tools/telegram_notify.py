#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patched: compute metrics from trades.csv when metrics.json is missing/incomplete,
so Telegram never shows 0 trades / 0 ROI with a rising equity curve.
"""

import os, sys, argparse, json, time, glob
import requests
import pandas as pd
import numpy as np

def eprint(*a): print(*a, file=sys.stderr)

def read_textfile(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def _metrics_from_trades(trades_csv, initial_capital=100000.0):
    try:
        df = pd.read_csv(trades_csv)
        if df.empty:
            return {}
        # Find PnL column
        pnl_col = next((c for c in df.columns if "pnl" in c.lower() or "profit" in c.lower()), None)
        if pnl_col is None:
            return {}
        # Closed trades only (if status exists)
        if "status" in df.columns:
            closed = df[df["status"].astype(str).str.lower().isin(["closed","filled","exit","complete"])].copy()
            if closed.empty:
                closed = df.copy()
        else:
            closed = df.copy()

        trades = int(len(closed))
        wins = int((closed[pnl_col] > 0).sum())
        losses = int((closed[pnl_col] < 0).sum())
        winrate = 100 * wins / trades if trades else 0.0

        equity = initial_capital + closed[pnl_col].cumsum()
        if trades:
            final_capital = float(equity.iloc[-1])
        else:
            final_capital = float(initial_capital)
        roi_pct = 100 * (final_capital - initial_capital) / initial_capital

        peak = equity.cummax()
        dd = (equity - peak) / peak
        maxdd_pct = 100 * float(dd.min()) if trades else 0.0
        time_dd_bars = int((dd < 0).sum()) if trades else 0

        rets = closed[pnl_col] / float(initial_capital)
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(len(rets))) if trades and rets.std() != 0 else 0.0

        total_profit = float(closed.loc[closed[pnl_col] > 0, pnl_col].sum())
        total_loss = float(-closed.loc[closed[pnl_col] < 0, pnl_col].sum())
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        else:
            profit_factor = float("inf") if total_profit > 0 else 0.0

        avg_win = float(closed.loc[closed[pnl_col] > 0, pnl_col].mean()) if wins else 0.0
        avg_loss = float(-closed.loc[closed[pnl_col] < 0, pnl_col].mean()) if losses else 0.0
        rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        return {
            "trades": trades,
            "win_rate": round(winrate, 2),
            "roi_pct": round(roi_pct, 2),
            "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else "Inf",
            "rr": round(rr, 2),
            "max_dd_pct": round(maxdd_pct, 2),
            "time_dd_bars": time_dd_bars,
            "sharpe_ratio": round(sharpe, 2),
            "final_capital": round(final_capital, 2),
        }
    except Exception as ex:
        eprint("WARN: _metrics_from_trades failed:", ex)
        return {}

def build_summary(rep_dir: str) -> str:
    # 0) Try metrics.json
    metrics_path = os.path.join(rep_dir, "metrics.json")
    if os.path.isfile(metrics_path):
        try:
            m = json.load(open(metrics_path, "r", encoding="utf-8"))
            # If trades missing/zero → try to augment from trades.csv
            if int(m.get("trades", 0) or 0) == 0:
                trades_csv = os.path.join(rep_dir, "trades.csv")
                if os.path.isfile(trades_csv):
                    m2 = _metrics_from_trades(trades_csv)
                    if m2:
                        m.update(m2)
            return (
                "📊 <b>Backtest Summary</b>\n"
                f"• Trades: <b>{m.get('trades', 0)}</b>\n"
                f"• Win-rate: <b>{m.get('win_rate', 0)}%</b>\n"
                f"• ROI: <b>{m.get('roi_pct', 0)}%</b>\n"
                f"• Profit Factor: <b>{m.get('profit_factor', 0)}</b>\n"
                f"• R:R: <b>{m.get('rr', 0)}</b>\n"
                f"• Max DD: <b>{m.get('max_dd_pct', 0)}%</b>\n"
                f"• Time DD (bars): <b>{m.get('time_dd_bars', 0)}</b>\n"
                f"• Sharpe: <b>{m.get('sharpe_ratio', 0)}</b>"
            )
        except Exception as ex:
            eprint("WARN: metrics.json parse failed:", ex)

    # 1) Try trades.csv directly (strong fallback)
    trades_csv = os.path.join(rep_dir, "trades.csv")
    if os.path.isfile(trades_csv):
        m = _metrics_from_trades(trades_csv)
        if m:
            return (
                "📊 <b>Backtest Summary</b>\n"
                f"• Trades: <b>{m.get('trades', 0)}</b>\n"
                f"• Win-rate: <b>{m.get('win_rate', 0)}%</b>\n"
                f"• ROI: <b>{m.get('roi_pct', 0)}%</b>\n"
                f"• Profit Factor: <b>{m.get('profit_factor', 0)}</b>\n"
                f"• R:R: <b>{m.get('rr', 0)}</b>\n"
                f"• Max DD: <b>{m.get('max_dd_pct', 0)}%</b>\n"
                f"• Time DD (bars): <b>{m.get('time_dd_bars', 0)}</b>\n"
                f"• Sharpe: <b>{m.get('sharpe_ratio', 0)}</b>"
            )

    # 2) summary.json (legacy)
    js_path = os.path.join(rep_dir, "summary.json")
    if os.path.isfile(js_path):
        try:
            data = json.load(open(js_path, "r", encoding="utf-8"))
            trades  = data.get("trades", 0)
            winrate = data.get("win_rate", data.get("winrate", 0))
            roi     = data.get("roi_pct", data.get("roi", 0))
            pf      = data.get("profit_factor", 0)
            rr      = data.get("rr", data.get("r_r", 0))
            maxdd   = data.get("max_dd_pct", data.get("max_dd", 0))
            tdd     = data.get("time_dd_bars", data.get("time_dd", 0))
            sharpe  = data.get("sharpe", data.get("sharpe_ratio", 0))
            return (
                f"📊 <b>Backtest Summary</b>\n"
                f"• Trades: <b>{trades}</b>\n"
                f"• Win-rate: <b>{round(float(winrate),2)}%</b>\n"
                f"• ROI: <b>{round(float(roi),2)}%</b>\n"
                f"• Profit Factor: <b>{round(float(pf),2)}</b>\n"
                f"• R:R: <b>{round(float(rr),2)}</b>\n"
                f"• Max DD: <b>{round(float(maxdd),2)}%</b>\n"
                f"• Time DD (bars): <b>{int(tdd)}</b>\n"
                f"• Sharpe: <b>{round(float(sharpe),2)}</b>"
            )
        except Exception as ex:
            eprint("WARN: summary.json parse failed:", ex)

    # 3) summary.txt
    txt_path = os.path.join(rep_dir, "summary.txt")
    if os.path.isfile(txt_path):
        txt = read_textfile(txt_path)
        if txt:
            return f"📊 <b>Backtest Summary</b>\n{txt}"

    return "📊 <b>Backtest Summary</b>\nNote: No metrics found."

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

    # 0) message
    msg = build_summary(rep_dir)
    msg = f"<b>{title}</b>\n\n{msg}"

    # 1) send message
    send_message(args.token, args.chat, msg)

    # 2) send images
    pics = sorted(glob.glob(os.path.join(rep_dir, "*.png")))
    if not pics:
        pics = sorted(glob.glob(os.path.join(rep_dir, "*.jpg")))

    for p in pics:
        try:
            send_photo(args.token, args.chat, p, caption=os.path.basename(p))
            time.sleep(0.4)
        except Exception as ex:
            eprint("WARN: sending photo failed for", p, ex)

    print("✅ Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
