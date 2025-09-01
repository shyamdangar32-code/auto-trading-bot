# tools/ensure_report.py
# -----------------------------------------------------------
# Minimal-report helper used by GitHub Actions steps.
# Ensures that downstream steps (artifacts upload, telegram)
# never fail even when real backtest produces no data.
# -----------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


__all__ = ["ensure_report"]


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def ensure_report(outdir: str = "./reports", note: str = "No summary") -> None:
    """
    Create a minimal set of report files expected by the workflow:

      - reports/metrics.json
      - reports/trades.csv
      - logs/summary.txt

    Safe to call multiple times. Does NOT overwrite real files
    if they already exist with content.
    """
    out_dir = Path(outdir)
    logs_dir = Path("./logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1) metrics.json (only create if missing/empty)
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        metrics = {
            "trades": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "profit_factor": 0.0,
            "rr": 0.0,
            "max_dd_pct": 0.0,
            "time_dd_bars": 0,
            "sharpe": 0.0,
            "note": note,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        _safe_write_json(metrics_path, metrics)

    # 2) trades.csv header (only if file missing/empty)
    trades_path = out_dir / "trades.csv"
    if not trades_path.exists() or trades_path.stat().st_size == 0:
        _safe_write_text(trades_path, "time,signal,price,qty,pnl\n")

    # 3) logs/summary.txt (always refresh with latest note)
    summary_path = logs_dir / "summary.txt"
    _safe_write_text(summary_path, f"Backtest Summary: {note}\n")

    print(
        f"ensure_report: ready → {metrics_path}, {trades_path}, {summary_path}"
    )
