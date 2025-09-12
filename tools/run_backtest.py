# tools/run_backtest.py
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import pandas as pd

try:
    import yaml  # optional
except Exception:
    yaml = None

from bot.data_io import get_zerodha_ohlc
from bot.strategy import prepare_signals
from bot.backtest import run_backtest, save_reports


# ---------------- CLI ---------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run backtest (LEGACY by default; profiles opt-in)"
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--underlying", required=True)
    p.add_argument("--start", required=True)                # YYYY-MM-DD
    p.add_argument("--end", required=True)                  # YYYY-MM-DD
    p.add_argument("--interval", default="1m")
    p.add_argument("--capital_rs", type=float, default=100000.0)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_block", default="backtest_loose",
                   choices=["backtest_loose","backtest_medium","backtest_strict"])
    # NEW: how much of YAML to honor
    p.add_argument("--profile_mode", default="off",
                   choices=["off","safe","full"],
                   help=("off=legacy (no overrides), "
                         "safe=signals-only overrides, "
                         "full=signals+engine overrides"))
    return p.parse_args()


def _ensure_dirs(outdir: str) -> Path:
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    return p


def _safe_load_yaml(path: str) -> dict:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _deep_merge(a: dict, b: dict) -> dict:
    out = deepcopy(a) if isinstance(a, dict) else {}
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _block_to_profile(use_block: str) -> str:
    x = (use_block or "").strip().lower()
    if x.startswith("backtest_"):
        return x.split("backtest_", 1)[1] or "loose"
    return x or "loose"


# --------------- MAIN ---------------- #
def main() -> None:
    args = _parse_args()
    outdir = _ensure_dirs(args.outdir)

    print(f"🧾 Fetching Zerodha OHLC: symbol={args.underlying} interval={args.interval} "
          f"{args.start} -> {args.end}")
    prices = get_zerodha_ohlc(args.underlying, args.start, args.end, args.interval)
    if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
        (outdir / "metrics.json").write_text(json.dumps({"error":"no_data"}), encoding="utf-8")
        raise SystemExit("No OHLC data returned — check Zerodha credentials or date range.")

    # Runtime trade config (same as legacy)
    base_runtime_cfg = {
        "capital_rs": float(args.capital_rs),
        "order_qty": int(args.order_qty),
        "paper_trading": True,
        "live_trading": False,
    }
    prof = _block_to_profile(args.use_block)
    print(f"⚙️  Trade config: {{'order_qty': {base_runtime_cfg['order_qty']}, "
          f"'capital_rs': {base_runtime_cfg['capital_rs']}}}")
    print(f"🧱 Using profile: {prof} | mode: {args.profile_mode}")

    # Legacy signal plan (keeps strategy defaults; default strategy = ema_rsi_adx)
    compat_plan = base_runtime_cfg | {"backtest": {}, "market_tz": "Asia/Kolkata"}

    # Load YAML (optional)
    cfg_yaml = _safe_load_yaml(args.config)
    base_block = cfg_yaml.get("backtest") or {}
    prof_block = cfg_yaml.get(args.use_block) or {}
    merged = _deep_merge(base_block, prof_block) if (base_block or prof_block) else {}

    # Whitelists
    SAFE_SIGNAL_KEYS = {
        "risk_perc", "atr_len", "atr_mult_sl", "atr_mult_tp",
        "allow_shorts", "trend_filter", "htf_minutes",
        # NOTE: we intentionally DO NOT honor 'strategy' unless you use --profile_mode full
    }
    SAFE_ENGINE_KEYS = {
        "min_hold_bars", "cooldown_bars", "trail_after_hold",
        "atr_mult_sl", "atr_mult_tp",
    }

    # -------- Build signals DF -------- #
    if args.profile_mode == "off":
        # pure legacy: ignore YAML completely
        plan_for_signals = compat_plan
    elif args.profile_mode == "safe":
        # only allow benign signal tweaks
        plan_for_signals = compat_plan | {k: v for k, v in merged.items() if k in SAFE_SIGNAL_KEYS}
    else:  # full
        # allow strategy switch too if provided
        plan_for_signals = compat_plan | merged

    df = prepare_signals(prices, plan_for_signals)

    # -------- Engine cfg for backtest -------- #
    if args.profile_mode == "full":
        engine_cfg = {"backtest": deepcopy(base_runtime_cfg)} | deepcopy(base_runtime_cfg)
        engine_cfg["backtest"].update({k: v for k, v in merged.items() if k in SAFE_ENGINE_KEYS or k in {"session_end","market_tz","tz"}})
    else:
        # legacy engine (no min_hold/cooldown)
        engine_cfg = {"backtest": deepcopy(base_runtime_cfg)} | deepcopy(base_runtime_cfg)

    summary, trades_df, equity_ser = run_backtest(
        df_in=df,
        cfg=engine_cfg,
        use_block=args.use_block,
    )

    save_reports(outdir=outdir, summary=summary, trades_df=trades_df, equity_ser=equity_ser)
    print("✅ Backtest finished & reports saved.")


if __name__ == "__main__":
    main()
