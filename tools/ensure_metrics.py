# tools/ensure_metrics.py
from __future__ import annotations
import json, pathlib, csv, argparse, datetime as dt, os

def is_nonempty(p: pathlib.Path) -> bool:
    try:
        return p.exists() and p.stat().st_size > 10
    except Exception:
        return False

def write_json(p: pathlib.Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def write_empty_signals_csv(p: pathlib.Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    headers = ["timestamp","action","symbol","price","qty","pnl"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(headers)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports")
    ap.add_argument("--note", default="No signals generated today.")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    rdir = pathlib.Path(args.dir)
    rdir.mkdir(parents=True, exist_ok=True)

    metrics_p = rdir / "metrics.json"
    latest_p  = rdir / "latest.json"
    signals_p = rdir / "latest_signals.csv"

    need_fallback = not any(map(is_nonempty, [metrics_p, latest_p, signals_p]))

    if not is_nonempty(signals_p):
        write_empty_signals_csv(signals_p)

    report_title = args.title or os.getenv("REPORT_TITLE") or None

    if not is_nonempty(metrics_p):
        metrics = {
            "ts": dt.datetime.utcnow().isoformat() + "Z",
            "n_trades": 0,
            "win_rate": 0.0,
            "roi_pct": 0.0,
            "profit_factor": 0.0,
            "rr": 0.0,
            "max_dd_pct": 0.0,
            "time_dd_bars": 0,
            "sharpe_ratio": 0.0,
            "note": args.note,
        }
        if report_title:
            metrics["report_title"] = report_title
        write_json(metrics_p, metrics)

    if not is_nonempty(latest_p):
        latest = {
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "summary": args.note,
            "config": {},
        }
        write_json(latest_p, latest)

    print("✅ ensure_metrics: reports are present.",
          f"(fallback created: {need_fallback})")

if __name__ == "__main__":
    main()
