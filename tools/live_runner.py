#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-Live runner (NO real orders).
- Re-uses tools.run_backtest on a rolling window.
- Detects new closed trades → Telegram + append live CSV.
- Detects Zerodha token expiry → Telegram warning + retry/backoff.
- At market close (~15:30 IST) → send Daily Summary, write summary file, exit.
"""

import os, sys, time, json, csv, argparse, shutil, subprocess, datetime as dt
from pathlib import Path
import requests

# ---------------- Utilities ----------------

def eprint(*a): print(*a, file=sys.stderr)

def ist_now():
    return dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)

def market_open(now_ist: dt.datetime):
    if now_ist.weekday() >= 5:  # Sat/Sun
        return False
    t = now_ist.time()
    return (dt.time(9,15) <= t <= dt.time(15,30))

def after_close(now_ist: dt.datetime):
    return now_ist.time() >= dt.time(15,31)

def run_backtest_once(args, start_date: str, end_date: str, outdir: str):
    """Invoke backtest; return (Path, err_tag)"""
    if os.path.exists(outdir):
        shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "tools.run_backtest",
        "--underlying", args.underlying,
        "--start", start_date,
        "--end",   end_date,
        "--interval", args.interval,
        "--capital_rs", str(args.capital_rs),
        "--order_qty",  str(args.order_qty),
        "--outdir", outdir,
        "--use_block", f"backtest_{args.profile}",
        "--live_mode",
        "--warmup_days", str(args.lookback_days),
    ]
    eprint("▶ Running backtest:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    err_tag = None
    if proc.returncode != 0:
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        eprint("❌ backtest failed")
        eprint(out[:2000])
        low = out.lower()
        if ("token" in low and "expire" in low) or ("tokenexception" in low) or ("401" in low) \
           or ("unauthorized" in low) or ("forbidden" in low) or ("403" in low):
            err_tag = "TOKEN"
        else:
            err_tag = "OTHER"
    return Path(outdir), err_tag

def read_trades_count(trades_csv: Path) -> int:
    if not trades_csv.exists(): return 0
    with trades_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r, None)
        return sum(1 for _ in r)

def tg_send(token, chat, text):
    if not token or not chat: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": text,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=20
        )
    except Exception as ex:
        eprint("WARN: Telegram send failed:", ex)

def write_live_log_row(csv_path: Path, row: dict):
    newfile = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newfile: w.writeheader()
        w.writerow(row)

def read_metrics(rep: Path):
    m = {}
    p = rep / "metrics.json"
    if p.exists():
        try: m = json.loads(p.read_text(encoding="utf-8"))
        except Exception as ex: eprint("WARN: metrics read:", ex)
    return {
        "trades":        int(m.get("trades", m.get("n_trades", 0) or 0)),
        "win_rate":      float(m.get("win_rate", m.get("winrate", 0) or 0)),
        "roi_pct":       float(m.get("roi_pct", m.get("ROI", 0) or 0)),
        "profit_factor": m.get("profit_factor", 0),
        "rr":            float(m.get("rr", m.get("R_R", 0) or 0)),
        "max_dd_pct":    float(m.get("max_dd_pct", m.get("max_dd_perc", 0) or 0)),
        "time_dd_bars":  int(m.get("time_dd_bars", m.get("time_dd", 0) or 0)),
        "sharpe_ratio":  float(m.get("sharpe_ratio", m.get("sharpe", 0) or 0)),
        "final_capital": float(m.get("final_capital", m.get("FinalCapital", 0) or 0)),
    }

def write_summary_file(base_dir: Path, metrics: dict, args):
    base_dir.mkdir(parents=True, exist_ok=True)
    name = f"summary_{ist_now():%Y-%m-%d}.md"
    md = base_dir / name
    lines = [
        f"# Paper Live Summary — {args.underlying} • {args.interval} • {args.profile}",
        "",
        f"- **Trades**: {metrics['trades']}",
        f"- **Win-rate**: {metrics['win_rate']:.2f}%",
        f"- **ROI**: {metrics['roi_pct']:.2f}%",
        f"- **Profit Factor**: {metrics['profit_factor']}",
        f"- **R:R**: {metrics['rr']:.2f}",
        f"- **Max DD**: {metrics['max_dd_pct']:.2f}%",
        f"- **Time DD (bars)**: {metrics['time_dd_bars']}",
        f"- **Sharpe**: {metrics['sharpe_ratio']:.2f}",
        f"- **Final Capital**: {metrics['final_capital']}",
        ""
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return md

# ---------------- Main loop ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paper"], default="paper")
    ap.add_argument("--underlying", required=True)
    ap.add_argument("--interval", required=True)
    ap.add_argument("--profile", choices=["loose","medium","strict"], required=True)
    ap.add_argument("--capital_rs", type=float, required=True)
    ap.add_argument("--order_qty", type=int, required=True)
    ap.add_argument("--lookback_days", type=int, default=5)
    ap.add_argument("--poll_seconds", type=int, default=60)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--telegram_token", default="")
    ap.add_argument("--telegram_chat",  default="")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir = outdir.parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    trades_live_csv = outdir.parent / f"trades_live_{args.profile}.csv"
    state_path = outdir.parent / f".state_{args.profile}.json"
    state = {"last_trades_count": 0, "token_warned": False, "summary_sent": False}
    if state_path.exists():
        try: state.update(json.loads(state_path.read_text(encoding="utf-8")))
        except Exception as ex: eprint("WARN: state read:", ex)

    tg_prefix = f"🟩 <b>PAPER</b> • {args.underlying} • {args.interval} • {args.profile}"
    tg_send(args.telegram_token, args.telegram_chat,
            f"{tg_prefix}\nStarted at {ist_now():%Y-%m-%d %H:%M IST}\n"
            f"Polling every {args.poll_seconds}s, lookback {args.lookback_days}d")

    while True:
        now = ist_now()

        # End-of-day summary & exit
        if after_close(now) and not state.get("summary_sent", False):
            metrics = read_metrics(outdir)
            msg = (f"{tg_prefix}\n"
                   f"📊 <b>Daily Summary</b>\n"
                   f"Trades {metrics['trades']}, Win% {metrics['win_rate']:.2f}, ROI% {metrics['roi_pct']:.2f}\n"
                   f"PF {metrics['profit_factor']}, R:R {metrics['rr']:.2f}, MaxDD {metrics['max_dd_pct']:.2f}%, Sharpe {metrics['sharpe_ratio']:.2f}")
            tg_send(args.telegram_token, args.telegram_chat, msg)
            write_summary_file(outdir.parent, metrics, args)
            state["summary_sent"] = True
            state_path.write_text(json.dumps(state), encoding="utf-8")
            break

        if not market_open(now):
            time.sleep(max(args.poll_seconds, 60))
            continue

        start = (now.date() - dt.timedelta(days=args.lookback_days)).isoformat()
        end   = now.date().isoformat()

        rep, err = run_backtest_once(args, start, end, str(outdir))
        if err == "TOKEN":
            if not state.get("token_warned", False):
                tg_send(args.telegram_token, args.telegram_chat,
                        f"{tg_prefix}\n⚠️ Zerodha access token seems <b>expired</b>. Update token in GitHub Secrets and restart.\nRetrying…")
                state["token_warned"] = True
                state_path.write_text(json.dumps(state), encoding="utf-8")
            time.sleep(max(60, args.poll_seconds))
            continue
        elif err == "OTHER":
            time.sleep(max(30, args.poll_seconds))
            continue
        else:
            if state.get("token_warned", False):
                tg_send(args.telegram_token, args.telegram_chat,
                        f"{tg_prefix}\n✅ Token OK again. Resumed.")
                state["token_warned"] = False
                state_path.write_text(json.dumps(state), encoding="utf-8")

        trades_csv = Path(outdir) / "trades.csv"
        new_count = read_trades_count(trades_csv)
        if new_count > state.get("last_trades_count", 0):
            delta = new_count - state.get("last_trades_count", 0)
            metrics = read_metrics(rep)
            msg = (f"{tg_prefix}\n"
                   f"📈 New closed trades: <b>{delta}</b> (total {new_count})\n"
                   f"Win% <b>{metrics['win_rate']:.2f}</b> | ROI% <b>{metrics['roi_pct']:.2f}</b> | PF <b>{metrics['profit_factor']}</b> | R:R <b>{metrics['rr']:.2f}</b>\n"
                   f"MaxDD% <b>{metrics['max_dd_pct']:.2f}</b> | Sharpe <b>{metrics['sharpe_ratio']:.2f}</b>\n"
                   f"{now:%Y-%m-%d %H:%M IST}")
            tg_send(args.telegram_token, args.telegram_chat, msg)

            row = {
                "ts_ist": now.strftime("%Y-%m-%d %H:%M:%S"),
                "underlying": args.underlying,
                "interval": args.interval,
                "profile": args.profile,
                "new_trades": delta,
                "total_trades": new_count,
                "win_rate": metrics["win_rate"],
                "roi_pct": metrics["roi_pct"],
                "profit_factor": metrics["profit_factor"],
                "rr": metrics["rr"],
                "max_dd_pct": metrics["max_dd_pct"],
                "sharpe": metrics["sharpe_ratio"],
                "reports_dir": str(rep)
            }
            write_live_log_row(trades_live_csv, row)

            state["last_trades_count"] = new_count
            state_path.write_text(json.dumps(state), encoding="utf-8")

        time.sleep(max(10, args.poll_seconds))

    return 0

if __name__ == "__main__":
    sys.exit(main())
