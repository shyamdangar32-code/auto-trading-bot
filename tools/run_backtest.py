# tools/run_backtest.py  (FINAL — base+profile deep-merge, CLI overrides, safe logs)
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from bot.data_io import get_zerodha_ohlc
from bot.strategy import prepare_signals
from bot.backtest import run_backtest, save_reports


# ---------------------- CLI ---------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backtest with config profiles (safe deep-merge)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--underlying", required=True)        # e.g. NIFTY
    p.add_argument("--start", required=True)             # YYYY-MM-DD
    p.add_argument("--end", required=True)               # YYYY-MM-DD
    p.add_argument("--interval", default="1m")           # 1m/3m/5m/15m/day/...
    # optional runtime overrides
    p.add_argument("--capital_rs", type=float)
    p.add_argument("--order_qty", type=int)
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--use_block",
        default="backtest_loose",
        choices=["backtest_loose", "backtest_medium", "backtest_strict"],
        help="Profile block to overlay on top of the base 'backtest' block",
    )
    return p.parse_args()


def _ensure_dirs(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    return p


# ---------------------- Config helpers ---------------------- #
def _deep_merge(a: dict, b: dict) -> dict:
    """Deep merge: returns copy of a overlaid with b (b overrides a)."""
    out = deepcopy(a) if isinstance(a, dict) else {}
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml did not parse into a mapping/dict")
    return data


def _build_plan_and_runtime(cfg: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    """
    Returns (runtime_cfg, plan)
      plan  -> backtest base + profile deep-merge + CLI overrides (used by prepare_signals)
      runtime_cfg -> full cfg passed into run_backtest (keeps all blocks)
    """
    tz = cfg.get("tz", "Asia/Kolkata")

    base = cfg.get("backtest", {}) or {}
    profile = cfg.get(args.use_block, {}) or {}

    # 1) deep-merge base + profile  (profile only overrides provided keys)
    plan = _deep_merge(base, profile)

    # 2) Inherit some important top-level defaults if missing
    # (avoid accidental drops causing behaviour drift)
    for k in ("strategy", "ema_fast", "ema_slow", "rsi_len", "rsi_buy", "rsi_sell",
              "atr_len", "atr_mult_sl", "atr_mult_tp", "risk_perc",
              "allow_shorts", "trend_filter", "htf_minutes",
              "min_hold_bars", "cooldown_bars", "trail_after_hold"):
        if k in cfg and k not in plan:
            plan[k] = cfg[k]

    # Ensure TZ keys exist for backtest logic
    plan.setdefault("tz", tz)
    plan.setdefault("market_tz", tz)

    # 3) CLI overrides (highest priority, only if provided)
    if args.capital_rs is not None:
        plan["capital_rs"] = float(args.capital_rs)
    if args.order_qty is not None:
        plan["order_qty"] = int(args.order_qty)

    # runtime_cfg keeps original cfg + also reflect simple fallbacks
    runtime_cfg = deepcopy(cfg)
    if args.capital_rs is not None:
        runtime_cfg["capital_rs"] = float(args.capital_rs)
    if args.order_qty is not None:
        runtime_cfg["order_qty"] = int(args.order_qty)

    return runtime_cfg, plan


# ---------------------- Main ---------------------- #
def main() -> None:
    args = _parse_args()
    outdir = _ensure_dirs(args.outdir)

    # Load & merge configuration
    cfg = _load_config(args.config)
    runtime_cfg, plan = _build_plan_and_runtime(cfg, args)

    # Helpful logs
    print("────────────────────────────────────────────────")
    print(f"📁 Config: {args.config}")
    print(f"🧱 Profile: {args.use_block}")
    print(f"🕒 TZ: {plan.get('market_tz', 'Asia/Kolkata')}")
    print(f"⚙️  Strategy: {plan.get('strategy', 'ema_rsi_adx')}")
    print(f"💰 Capital: {plan.get('capital_rs')} | Qty: {plan.get('order_qty')}")
    print(f"🔢 Params: ema({plan.get('ema_fast')},{plan.get('ema_slow')}), "
          f"rsi_len={plan.get('rsi_len')}, rsi_buy={plan.get('rsi_buy')}, rsi_sell={plan.get('rsi_sell')}, "
          f"atr_len={plan.get('atr_len')}, SLxATR={plan.get('atr_mult_sl')}, TPxATR={plan.get('atr_mult_tp')}")
    print("────────────────────────────────────────────────")

    # Fetch OHLC
    print(f"🧾 Fetching Zerodha OHLC: symbol={args.underlying} interval={args.interval} {args.start} -> {args.end}")
    prices = get_zerodha_ohlc(args.underlying, args.start, args.end, args.interval)
    if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
        (Path(outdir) / "metrics.json").write_text(json.dumps({"error": "no_data"}), encoding="utf-8")
        raise SystemExit("No OHLC data returned — check Zerodha credentials, token, or date range.")

    # Build signals from merged plan
    df = prepare_signals(prices, plan)

    # Pass full cfg (so backtest can also read blocks) and explicitly specify profile used
    runtime_cfg["__profile_used__"] = args.use_block
    summary, trades_df, equity_ser = run_backtest(
        df=df,
        cfg=runtime_cfg,
        use_block=args.use_block,
    )

    # Save artifacts
    save_reports(
        outdir=outdir,
        summary=summary,
        trades_df=trades_df,
        equity_ser=equity_ser,
    )

    print("✅ Backtest finished & reports saved.")


if __name__ == "__main__":
    main()
