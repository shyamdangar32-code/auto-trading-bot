# tools/run_backtest.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from copy import deepcopy

import pandas as pd
import yaml

from bot.data_io import get_zerodha_ohlc
from bot.strategy import prepare_signals
from bot.backtest import run_backtest, save_reports


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backtest with config profiles")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--underlying", required=True)        # e.g. NIFTY
    p.add_argument("--start", required=True)             # YYYY-MM-DD
    p.add_argument("--end", required=True)               # YYYY-MM-DD
    p.add_argument("--interval", default="1m")           # 1m/3m/5m/15m/day/...
    p.add_argument("--capital_rs", type=float)           # override optional
    p.add_argument("--order_qty", type=int)              # override optional
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_block", default="backtest_loose", choices=[
        "backtest_loose", "backtest_medium", "backtest_strict"
    ])
    return p.parse_args()


def _ensure_dirs(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    return p


def _deep_merge(a: dict, b: dict) -> dict:
    """return deep-merged dict = a <- b (b overrides a)"""
    out = deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_runtime_cfg(cfg: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    """
    Returns:
      - runtime_cfg: full config passed to backtest (keeps blocks so bot/backtest.py can also read them)
      - plan: flattened plan for signal building (backtest + selected profile + CLI overrides)
    """
    tz = cfg.get("tz", "Asia/Kolkata")
    base_bt = cfg.get("backtest", {})                    # default backtest block
    prof_bt = cfg.get(args.use_block, {})                # selected profile override block

    # 1) merge defaults + profile
    plan = _deep_merge(base_bt, prof_bt)

    # 2) CLI overrides on top (only if provided)
    if args.capital_rs is not None:
        plan["capital_rs"] = float(args.capital_rs)
    if args.order_qty is not None:
        plan["order_qty"] = int(args.order_qty)

    # ensure tz present for backtest logic
    plan.setdefault("tz", tz)
    plan.setdefault("market_tz", tz)

    # runtime cfg keeps blocks so bot/backtest.py can merge again if it wants
    runtime_cfg = deepcopy(cfg)
    # also surface top-level commonly used keys as fallbacks
    if args.capital_rs is not None:
        runtime_cfg["capital_rs"] = float(args.capital_rs)
    if args.order_qty is not None:
        runtime_cfg["order_qty"] = int(args.order_qty)

    return runtime_cfg, plan


def main() -> None:
    args = _parse_args()
    outdir = _ensure_dirs(args.outdir)

    # ---- Load config + build plan/profile ----
    cfg = _load_config(args.config)
    runtime_cfg, plan = _build_runtime_cfg(cfg, args)

    print(f"⚙️ Profile: {args.use_block} | TZ={plan.get('market_tz','Asia/Kolkata')}")
    print(f"💰 Capital={plan.get('capital_rs')} | Qty={plan.get('order_qty')} | Strategy={plan.get('strategy','ema_rsi_adx')}")

    # ---- Fetch data ----
    print(f"🧾 Fetching Zerodha OHLC: symbol={args.underlying} interval={args.interval} {args.start} -> {args.end}")
    try:
        prices = get_zerodha_ohlc(args.underlying, args.start, args.end, args.interval)
    except Exception as e:
        raise RuntimeError(f"Failed fetching OHLC for {args.underlying}: {e}") from e

    if prices.empty:
        (outdir / "metrics.json").write_text(json.dumps({"error": "no_data"}), encoding="utf-8")
        raise SystemExit("No OHLC data returned — check Zerodha credentials, token, or date range.")

    # ---- Build signals (uses merged plan) ----
    df = prepare_signals(prices, plan)

    # ---- Run backtest (pass full cfg so bot/backtest.py may re-merge use_block) ----
    # Inject selected block label for clarity (not required, just for logging in downstream)
    runtime_cfg["__profile_used__"] = args.use_block

    summary, trades_df, equity_ser = run_backtest(
        df,
        runtime_cfg,
        use_block=args.use_block,
    )

    # ---- Save reports ----
    save_reports(
        outdir=outdir,
        summary=summary,
        trades_df=trades_df,
        equity_ser=equity_ser,
    )

    print("✅ Backtest finished & reports saved.")


if __name__ == "__main__":
    main()
