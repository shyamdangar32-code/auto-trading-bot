import os
import json
import argparse
from pathlib import Path

def ensure_minimal_report(report_dir: str):
    report_path = Path(report_dir) / "summary.json"

    if report_path.exists():
        print(f"[ensure_report] Found report at {report_path}")
        return

    print("[ensure_report] No summary.json found. Creating minimal report...")

    minimal_report = {
        "trades": 0,
        "win_rate": 0.0,
        "roi": 0.0,
        "profit_factor": 0.0,
        "rr": 0.0,
        "max_dd": 0.0,
        "time_dd": 0,
        "sharpe": 0.0,
        "note": "Auto-generated minimal report (no data)"
    }

    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(minimal_report, f, indent=2)

    print(f"[ensure_report] Minimal report written to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./reports")
    args = parser.parse_args()

    ensure_minimal_report(args.dir)
