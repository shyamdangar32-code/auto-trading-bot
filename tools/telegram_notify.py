# tools/telegram_notify.py
from __future__ import annotations
import os, json, argparse, requests, pathlib

SENSITIVE_KEYS = {"api", "token", "secret", "key", "password", "chat_id"}

def redact(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        lk = k.lower()
        out[k] = "****" if any(s in lk for s in SENSITIVE_KEYS) else v
    return out

def load_any_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def send_text(token: str, chat_id: str, text: str) -> None:
    requests.get(
        f"https://api.telegram.org/bot{token}/sendMessage",
        params={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        timeout=10,
    )

def send_file(token: str, chat_id: str, fpath: str, caption: str = "") -> None:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with open(fpath, "rb") as fh:
        files = {"document": fh}
        data = {"chat_id": chat_id, "caption": caption}
        requests.post(url, files=files, data=data, timeout=30)

def pick_title(metrics: dict, cli_title: str | None) -> str:
    # Priority: CLI --title > ENV REPORT_TITLE > metrics.report_title > workflow name heuristic > default
    if cli_title:
        return cli_title
    if os.getenv("REPORT_TITLE"):
        return os.getenv("REPORT_TITLE").strip()
    if isinstance(metrics, dict) and metrics.get("report_title"):
        return str(metrics["report_title"])
    wf = (os.getenv("GITHUB_WORKFLOW") or "").lower()
    if "intraday" in wf or "paper" in wf:
        return "Intraday Summary"
    if "backtest" in wf:
        return "Backtest Summary"
    return "Run Summary"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports", help="reports directory")
    ap.add_argument("--title", default=None, help="override title for Telegram card")
    args = ap.parse_args()

    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat  = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat:
        print("ℹ️  TELEGRAM_* not set; skipping notify.")
        return

    rdir = pathlib.Path(args.dir)
    metrics = load_any_json(str(rdir / "metrics.json"))
    legacy  = load_any_json(str(rdir / "latest.json"))

    title = pick_title(metrics, args.title)

    # Build summary text
    if metrics:
        m = metrics
        text = (
            f"📊 <b>{title}</b>\n"
            f"• Trades: <b>{m.get('n_trades', 0)}</b>\n"
            f"• Win-rate: <b>{m.get('win_rate', 0)}%</b>\n"
            f"• ROI: <b>{m.get('roi_pct', 0)}%</b>\n"
            f"• Profit Factor: <b>{m.get('profit_factor', 0)}</b>\n"
            f"• R:R: <b>{m.get('rr', 0)}</b>\n"
            f"• Max DD: <b>{m.get('max_dd_pct', 0)}%</b>\n"
            f"• Time DD (bars): <b>{m.get('time_dd_bars', 0)}</b>\n"
            f"• Sharpe: <b>{m.get('sharpe_ratio', 0)}</b>\n"
        )
        if m.get("note"):
            text += f"• Note: <i>{m['note']}</i>\n"
    else:
        text = f"📊 <b>{title}</b>\n"
        if legacy:
            cfg = redact(legacy.get("config", {}))
            meta = {k:v for k,v in legacy.items() if k != "config"}
            text += "\n".join([f"• {k}: <b>{v}</b>" for k,v in meta.items()])
            if cfg:
                text += "\n\n<b>Config</b>\n" + "\n".join([f"• {k}: {v}" for k,v in cfg.items()])
        else:
            text += "No summary available."

    # Send text
    send_text(token, chat, text)

    # Attach charts if present
    eq_png = rdir / "equity_curve.png"
    dd_png = rdir / "drawdown.png"
    if eq_png.exists():
        send_file(token, chat, str(eq_png), "Equity curve")
    if dd_png.exists():
        send_file(token, chat, str(dd_png), "Drawdown")

if __name__ == "__main__":
    main()
