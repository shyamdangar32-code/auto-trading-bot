# tools/run_backtest.py  (COMPAT + SAFE OVERRIDE)
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

# optional: only if config.yaml is present
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from bot.data_io import get_zerodha_ohlc
from bot.strategy import prepare_signals
from bot.backtest import run_backtest, save_reports


# -------------------- CLI -------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backtest (compat baseline + safe profile overrides)")
    p.add_argument("--config", default="config.yaml")     # optional; if missing, baseline path
    p.add_argument("--underlying", required=True)         # e.g., NIFTY
    p.add_argument("--start", required=True)              # YYYY-MM-DD
    p.add_argument("--end", required=True)                # YYYY-MM-DD
    p.add_argument("--interval", default="1m")            # 1m/3m/5m/15m/day/...
    p.add_argument("--capital_rs", type=float, default=100000.0)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_block", default="backtest_loose",
                   choices=["backtest_loose", "backtest_medium", "backtest_strict"])
    return p.parse_args()


def _ensure_dirs(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
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
    except FileNotFoundError:
        return {}
    except Exception:
        # on any parse error, fall back to baseline
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


# -------------------- Main -------------------- #
def main() -> None:
    args = _parse_args()
    outdir = _ensure_dirs(args.outdir)

    print(f"🧾 Fetching Zerodha OHLC: symbol={args.underlying} interval={args.interval} {args.start} -> {args.end}")
    prices = get_zerodha_ohlc(args.underlying, args.start, args.end, args.interval)

    if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
        (outdir / "metrics.json").write_text(json.dumps({"error": "no_data"}), encoding="utf-8")
        raise SystemExit("No OHLC data returned — check Zerodha credentials, token, or date range.")

    # -------- Baseline runtime config (exactly like your old file) -------- #
    # This preserves the previously good results.
    base_runtime_cfg = {
        "capital_rs": float(args.capital_rs),
        "order_qty": int(args.order_qty),
        "paper_trading": True,
        "live_trading": False,
    }

    profile_name = _block_to_profile(args.use_block)
    print(f"⚙️  Trade config: {{'order_qty': {base_runtime_cfg['order_qty']}, "
          f"'capital_rs': {base_runtime_cfg['capital_rs']}}}")
    print(f"🧱 Using profile: {profile_name}")

    # --------- Baseline plan for signals (keeps strategy defaults intact) --------- #
    # Exactly like the old behaviour: no strategy params pulled from YAML unless whitelisted below.
    compat_plan = base_runtime_cfg | {"backtest": {}, "market_tz": "Asia/Kolkata"}

    # --------- OPTIONAL: Safe, minimal overrides from config.yaml --------- #
    # Only allow these keys to override (so performance doesn't collapse).
    SAFE_SIGNAL_KEYS = {
        # risk sizing / exits
        "risk_perc", "atr_len", "atr_mult_sl", "atr_mult_tp",
        # behaviour controls
        "allow_shorts", "trend_filter", "htf_minutes",
    }
    SAFE_ENGINE_KEYS = {
        # backtest engine behaviour
        "min_hold_bars", "cooldown_bars", "trail_after_hold",
        "atr_mult_sl", "atr_mult_tp",
    }

    cfg_yaml = _safe_load_yaml(args.config)
    base_block = cfg_yaml.get("backtest") or {}
    prof_block = cfg_yaml.get(args.use_block) or {}
    merged_profile = _deep_merge(base_block, prof_block) if (base_block or prof_block) else {}

    # Apply only SAFE_SIGNAL_KEYS into plan (so core strategy defaults remain intact)
    safe_signal_overrides = {k: v for k, v in merged_profile.items() if k in SAFE_SIGNAL_KEYS}
    plan_for_signals = compat_plan | safe_signal_overrides

    # --------- Build df with signals --------- #
    df = prepare_signals(prices, plan_for_signals)

    # --------- Engine cfg (what backtest.py sees) --------- #
    # Old behaviour: {"backtest": base_runtime_cfg} | base_runtime_cfg
    engine_cfg = {"backtest": deepcopy(base_runtime_cfg)} | deepcopy(base_runtime_cfg)

    # From YAML, pass only SAFE_ENGINE_KEYS into engine backtest block
    if merged_profile:
        engine_cfg["backtest"].update({k: v for k, v in merged_profile.items() if k in SAFE_ENGINE_KEYS})

    # --------- Run backtest --------- #
    summary, trades_df, equity_ser = run_backtest(
        df_in=df,                 # NOTE: correct keyword
        cfg=engine_cfg,
        use_block=args.use_block, # kept for logging/compat, though engine_cfg already carries overrides
    )

    # --------- Save artifacts --------- #
    save_reports(
        outdir=outdir,
        summary=summary,
        trades_df=trades_df,
        equity_ser=equity_ser,
    )
    print("✅ Backtest finished & reports saved.")


if __name__ == "__main__":
    main()
