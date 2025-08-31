#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Send Telegram summary + charts for backtest/intraday runs.

Reads from a reports directory:
- Tries summary.json / summary.txt for headline metrics
- Attaches equity_curve.png and drawdown.png when present
Fails loudly with clear logs if Telegram API returns an error.
"""

import os, sys, argparse, json, time, glob
import requests

def eprint(*a):  # make logs visible in Actions
    print(*a, file=sys.stderr, flush=True)

def read_textfile(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as ex:
        return ""

def discover_summary(rep_dir):
    # Prefer JSON, else TXT, else craft minimal
    js_path = os.path.join(rep_dir, "summary.json")
    txt_path = os.path.join(rep_dir, "summary.txt")

    if os.path.isfile(js_path):
        try:
            data = json.load(open(js_path, "r", encoding="utf-8"))
            # very tolerant: pick common keys if present
            trades = data.get("trades", 0)
            winrate = data.get("win_rate", data.get("winrate", 0))
            roi = data.get("roi_pct", data.get("roi", 0))
            pf = data.get("profit_factor", 0)
            rr = data.get("rr", data.get("r_r", 0))
            maxdd = data.get("max_dd_pct", data.get("max_dd", 0))
            tdd = data.get("time_dd_bars", data.get("time_dd", 0))
            sharpe = data.get("sharpe", 0)

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

    if os.path.isfile(txt_path):
        txt = read_textfile(txt_path)
        if txt:
            return f"📊 <b>Backtest Summary</b>\n{txt}"

    # Minimal fallback if nothing available
    return "📊 <b>Backtest Summary</b>\nNote: No data; runner fallback."

def tg_send(token, method, payload=None, files=None):
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
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
    except requests.exceptions.RequestException as ex:
        eprint("❌ Telegram request failed:", ex)
        sys.exit(1)

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

    title = args.title.strip()
    headline = discover_summary(rep_dir)
    if title:
        headline = f"<b>{title}</b>\n" + headline

    print("ℹ️ Using reports dir:", rep_dir)
    print("ℹ️ Headline to send:\n", headline)

    # 1) Send headline message
    send_message(token, chat_id, headline)

    # 2) Attach charts if present
    pics = []
    for name in ["equity_curve.png", "drawdown.png"]:
        p = os.path.join(rep_dir, name)
        if os.path.isfile(p):
            pics.append(p)

    # If your runner writes PNGs with different names, attach any .png
    if not pics:
        pics = sorted(glob.glob(os.path.join(rep_dir, "*.png")))

    for p in pics:
        try:
            send_photo(token, chat_id, p, caption=os.path.basename(p))
            time.sleep(0.5)  # small spacing to respect rate limits
        except Exception as ex:
            eprint("WARN: sending photo failed for", p, ex)

    print("✅ Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
