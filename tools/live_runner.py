#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-Live runner (NO real orders).

⚠️ Strategy remains untouched:
We re-use your existing backtest entrypoint `tools.run_backtest`
exactly as-is on a rolling window (lookback_days), then read
the newly produced trades.csv to detect if a new trade got closed.
On change, we send a Telegram alert and append to trades_live.csv.

Requirements:
- tools.run_backtest must accept the same CLI flags you use in backtests.
- Your report folder will be recreated every poll (clean & re-run).
"""

import os, sys, time, json, csv, argparse, shutil, subprocess, datetime as dt
from pathlib import Path

# ---------------- Utilities ----------------

def eprint(*a): print(*a, file=sys.stderr)

def ist_now():
    # Compute India time without external libs
    # IST = UTC+5:30
    return dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)

def market_open(now_ist: dt.datetime):
    # Mon-Fri, 09:15–15:30 IST
    if now_ist.weekday() >= 5:   # 5=Sat, 6=Sun
        return False
    t = now_ist.time()
    return (dt.time(9,15) <= t <= dt.time(15,30))

def run_backtest_once(args, start_date: str, end_date: str, outdir: str) -> Path:
    """Invoke your existing backtest module to (re)build reports."""
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
        "--use_block", f"backtest_{args.profile}"
    ]
    eprint("▶ Running backtest:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        eprint("❌ backtest failed:", proc.stdout, proc.stderr)
        # keep going; caller can retry
    return Path(outdir)

def read_trades_count(trades_csv: Path) -> int:
    if not trades_csv.exists():
        return 0
    # count data rows (skip header)
    with trades_csv.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        n = sum(1 for _ in r)
    return n

def tg_send(token, chat, text):
    if not token or not chat:
        return
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": chat,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }, timeout=20)
    try:
        jr = resp.json()
    except Exception:
        jr = {"ok": False, "raw": resp.text}
    if not jr.get("ok", False):
        eprint("WARN: Telegram send failed:", jr)

def write_live_log_row(csv_path: Path, row: dict):
    newfile = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newfile: w.writeheader()
        w.writerow(row)

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
    logs_dir = outdir.parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    trades_live_csv = outdir.parent / f"trades_live_{args.profile}.csv"
    state_path = outdir.parent / f".state_{args.profile}.json"

    # Load state
    state = {"last_trades_count": 0}
    if state_path.exists():
        try:
            state.update(json.loads(state_path.read_text(encoding="utf-8")))
        except Exception as ex:
            eprint("WARN: state read:", ex)

    tg_prefix = f"🟩 <b>PAPER</b> • {args.underlying} • {args.interval} • {args.profile}"

    # First ping
    tg_send(args.telegram_token, args.telegram_chat,
            f"{tg_prefix}\nStarted at {ist_now():%Y-%m-%d %H:%M IST}\n"
            f"Polling every {args.poll_seconds}s, lookback {args.lookback_days}d")

    while True:
        now = ist_now()
        if not market_open(now):
            # sleep a little longer outside market hours
            time.sleep(max(args.poll_seconds, 60))
            continue

        start = (now.date() - dt.timedelta(days=args.lookback_days)).isoformat()
        end   = now.date().isoformat()

        # Rebuild reports with the same strategy (unchanged)
        rep = run_backtest_once(args, start, end, str(outdir))
        trades_csv = rep / "trades.csv"

        new_count = read_trades_count(trades_csv)
        if new_count > state.get("last_trades_count", 0):
            # A new trade got closed since last poll
            delta = new_count - state.get("last_trades_count", 0)

            # Pull summary from metrics.json if present
            metrics_path = rep / "metrics.json"
            metrics = {}
            try:
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as ex:
                eprint("WARN: metrics read:", ex)

            win_rate = metrics.get("win_rate", metrics.get("winrate", 0))
            roi = metrics.get("roi_pct", metrics.get("ROI", 0))
            pf  = metrics.get("profit_factor", 0)
            rr  = metrics.get("rr", metrics.get("R_R", 0))
            dd  = metrics.get("max_dd_pct", metrics.get("max_dd_perc", 0))
            shp = metrics.get("sharpe_ratio", metrics.get("sharpe", 0))

            msg = (f"{tg_prefix}\n"
                   f"📈 New closed trades: <b>{delta}</b> (total {new_count})\n"
                   f"Win% <b>{win_rate}</b> | ROI% <b>{roi}</b> | PF <b>{pf}</b> | R:R <b>{rr}</b>\n"
                   f"MaxDD% <b>{dd}</b> | Sharpe <b>{shp}</b>\n"
                   f"{now:%Y-%m-%d %H:%M IST}")
            tg_send(args.telegram_token, args.telegram_chat, msg)

            # Append to live log
            row = {
                "ts_ist": now.strftime("%Y-%m-%d %H:%M:%S"),
                "underlying": args.underlying,
                "interval": args.interval,
                "profile": args.profile,
                "new_trades": delta,
                "total_trades": new_count,
                "win_rate": win_rate,
                "roi_pct": roi,
                "profit_factor": pf,
                "rr": rr,
                "max_dd_pct": dd,
                "sharpe": shp,
                "reports_dir": str(rep)
            }
            write_live_log_row(trades_live_csv, row)

            # Save state
            state["last_trades_count"] = new_count
            state_path.write_text(json.dumps(state), encoding="utf-8")

        # small sleep until next poll
        time.sleep(max(10, args.poll_seconds))

    return 0

if __name__ == "__main__":
    sys.exit(main())
