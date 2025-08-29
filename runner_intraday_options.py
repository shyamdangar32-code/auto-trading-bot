#!/usr/bin/env python3
# runner_intraday_options.py
from __future__ import annotations
import os, sys, json, argparse, pathlib
import pandas as pd
import matplotlib.pyplot as plt

from bot.config import load_config, debug_fingerprint
from bot.data_io import prices as load_prices
from bot.evaluation import plot_equity_and_drawdown, write_quick_report

from core_backtest.index_level import run as run_index
from core_backtest.options_legs import simulate as run_options

def _save_metrics(out_dir: str, summary: dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary or {}, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--period", default="30d", help="download window if needed (Zerodha)")
    ap.add_argument("--interval", default="5m")
    args = ap.parse_args()

    cfg = load_config("config.yaml")
    out_dir = args.out_dir

    # ---- data ----
    df = load_prices(
        symbol=cfg["underlying"],
        period=args.period,
        interval=args.interval,
        zerodha_enabled=cfg.get("zerodha_enabled", True),
        zerodha_instrument_token=cfg.get("index_token"),
    )

    # normalize columns for our engines
    need = ["Open","High","Low","Close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"prices() missing cols: {missing}")

    # ---- run chosen mode ----
    mode = (cfg.get("mode") or "options").lower().strip()
    if mode == "index":
        summary, trades, equity = run_index(df[need], cfg)
        summary["mode"] = "index"
    elif mode == "options":
        summary, trades, equity = run_options(df[need], cfg)
        summary["mode"] = "options"
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    # ---- persist artifacts ----
    os.makedirs(out_dir, exist_ok=True)
    if trades is not None and not trades.empty:
        trades.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    if equity is not None and not equity.empty:
        equity.to_csv(os.path.join(out_dir, "equity.csv"), header=True)

    _save_metrics(out_dir, summary)
    if equity is not None and not equity.empty:
        plot_equity_and_drawdown(equity, out_dir)
    write_quick_report(summary, trades, out_dir)

    print("✅ Done. Artifacts in:", out_dir)
    print("🔒 Secrets:", debug_fingerprint(cfg))

if __name__ == "__main__":
    main()
