#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Send Telegram summary + charts for backtest/intraday runs.

Reads from a reports directory:
- Prefers metrics.json (our backtest writer)
- Falls back to summary.json / summary.txt
- Attaches equity_curve.png and drawdown.png
Fails loudly with clear logs if Telegram API returns an error.
"""

import os, sys, argparse, json, time, glob
import requests

def eprint(*a):
    print(*a, file=sys.stderr, flush=True)

def read_textfile(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

def discover_summary(rep_dir):
    """
    Try in this order:
      1) metrics.json  (keys from our backtest.py)
      2) summary.json  (older format)
      3) summary.txt   (plain text)
    """
    # 1) metrics.json
    mpath = os.path.join(rep_dir, "metrics.json")
    if os.path.isfile(mpath):
        try:
            m = json.load(open(mpath, "r", encoding="utf-8"))
            return (
                f"📊 <b>Backtest Summary</b>\n"
                f"• Trades: <b>{m.get('n_trades', 0)}</b>\n"
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
                f"• Time DD (bars): <b>{tdd}</b>\n"
                f"• Sharpe: <b>{round(float(sharpe),2)}</b>"
            )
        except Exception as ex:
            eprint("WARN: summary.json parse failed:", ex)

    # 3) summary.txt (legacy)
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
        return tg_send(
            token,
            "sendPhoto",
            payload={"chat_id": chat_id, "caption": caption or ""},
            files={"photo": (os.path.basename(path), f, "image/png")}
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", "--outdir", dest="rep_dir", default="./reports")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        eprint("❌ Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in environment.")
        sys.exit(1)

    rep_dir = os.path.abspath(args.rep_dir)
    if not os.path.isdir(rep_dir):
        eprint(f"❌ Reports directory not found: {rep_dir}")
        sys.exit(1)

    headline = discover_summary(rep_dir)
    title = args.title.strip()
    if title:
        headline = f"<b>{title}</b>\n" + headline

    print("ℹ️ Using reports dir:", rep_dir)
    print("ℹ️ Headline to send:\n", headline)

    # 1) message
    send_message(token, chat_id, headline)

    # 2) optional charts
    pics = []
    for name in ["equity_curve.png", "drawdown.png"]:
        p = os.path.join(rep_dir, name)
        if os.path.isfile(p):
            pics.append(p)
    if not pics:
        pics = sorted(glob.glob(os.path.join(rep_dir, "*.png")))

    for p in pics:
        try:
            send_photo(token, chat_id, p, caption=os.path.basename(p))
            time.sleep(0.4)
        except Exception as ex:
            eprint("WARN: sending photo failed for", p, ex)

    print("✅ Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
