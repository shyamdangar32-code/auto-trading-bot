# tools/run_backtest.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from bot.data_io import get_zerodha_ohlc
from bot.strategy import prepare_signals            # build signals (strategy chosen via plan['strategy'])
from bot.backtest import run_backtest, save_reports


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backtest via GitHub Actions / local")

    # === Your original args (kept) ===
    p.add_argument("--underlying", required=True)        # e.g. NIFTY
    p.add_argument("--start", required=True)             # YYYY-MM-DD
    p.add_argument("--end", required=True)               # YYYY-MM-DD
    p.add_argument("--interval", default="1m")           # 1m/3m/5m/15m/day/...
    p.add_argument("--capital_rs", type=float, default=100000.0)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_block", default="backtest_loose")  # backtest_loose/medium/strict

    return p.parse_args()


def _ensure_dirs(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    return p


def _load_cfg(path: str = "config.yaml") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def _block_to_profile(use_block: str) -> str:
    """Map 'backtest_loose' → 'loose' (used only for folder naming/logs)."""
    x = (use_block or "").strip().lower()
    if x.startswith("backtest_"):
        return x.split("backtest_", 1)[1] or "loose"
    return x or "loose"


def main() -> None:
    args = _parse_args()
    outdir = _ensure_dirs(args.outdir)

    # ---- Load config.yaml ----
    cfg_file = _load_cfg()

    tz = cfg_file.get("tz", "Asia/Kolkata")
    # printing range for log clarity
    print(f"🧾 Fetching Zerodha OHLC: symbol={args.underlying} interval={args.interval} {args.start} -> {args.end}")

    # ---- Fetch OHLC ----
    try:
        prices = get_zerodha_ohlc(args.underlying, args.start, args.end, args.interval)
    except Exception as e:
        raise RuntimeError(f"Failed fetching OHLC for {args.underlying}: {e}") from e

    if prices is None or len(prices) == 0:
        (outdir / "metrics.json").write_text(json.dumps({"error": "no_data"}), encoding="utf-8")
        raise SystemExit("No OHLC data returned — check Zerodha credentials, token, or date range.")

    # ---- Build PLAN for strategy (backtest defaults + profile overrides) ----
    use_block = args.use_block                      # e.g., backtest_loose
    profile_name = _block_to_profile(use_block)     # e.g., loose
    backtest_block = cfg_file.get("backtest") or {}
    profile_block = cfg_file.get(use_block) or {}   # block name is 'backtest_loose', etc.

    plan = {**backtest_block, **profile_block}
    # ensure core risk knobs are present (allow CLI override)
    plan.setdefault("capital_rs", float(args.capital_rs))
    plan.setdefault("order_qty", int(args.order_qty))

    # ---- Build ENGINE config (for run_backtest; can include global knobs too) ----
    cfg_all = {
        # prefer YAML top-level if present, then CLI as fallback
        "capital_rs": float(cfg_file.get("capital_rs", args.capital_rs)),
        "order_qty": int(cfg_file.get("order_qty", args.order_qty)),
        "out_dir": cfg_file.get("out_dir", "reports"),
        "paper_trading": cfg_file.get("paper_trading", True),
        "live_trading": cfg_file.get("live_trading", False),
        "tz": tz,
        # pass through full cfg for anything engine might need
        **cfg_file,
    }

    print(f"⚙️  Trade config: {{'order_qty': {cfg_all['order_qty']}, 'capital_rs': {cfg_all['capital_rs']}}}")
    print(f"🧱 Using profile: {profile_name}")

    # ---- Strategy signals (IMPORTANT: no 'profile' kwarg) ----
    df = prepare_signals(prices=prices, plan=plan)

    # ---- Run engine ----
    summary, trades_df, equity_ser = run_backtest(
        df_in=df,
        cfg=cfg_all,
        use_block=use_block,
    )

    # ---- Persist reports ----
    # folder like reports/loose, reports/medium, etc.
    reports_root = Path(cfg_all.get("out_dir", "reports"))
    final_outdir = reports_root / profile_name
    save_reports(
        outdir=final_outdir,
        summary=summary,
        trades_df=trades_df,
        equity_ser=equity_ser,
    )

    print("✅ Backtest finished & reports saved.")
    print("📈 Summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
