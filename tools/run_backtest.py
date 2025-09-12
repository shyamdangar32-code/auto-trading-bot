# tools/run_backtest.py
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# optional: YAML for config
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from bot.data_io import get_zerodha_ohlc
from bot.strategy import prepare_signals
from bot.backtest import run_backtest, save_reports


# ---------------- CLI ---------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run backtest (legacy-safe with profile overrides that never degrade)"
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--underlying", required=True)
    p.add_argument("--start", required=True)               # YYYY-MM-DD
    p.add_argument("--end", required=True)                 # YYYY-MM-DD
    p.add_argument("--interval", default="1m")             # 1m/3m/5m/15m/day/...
    p.add_argument("--capital_rs", type=float, default=100000.0)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_block", default="backtest_loose",
                   choices=["backtest_loose", "backtest_medium", "backtest_strict"])
    p.add_argument("--profile_mode", default="auto",
                   choices=["off", "safe", "full", "auto"],
                   help="off=legacy; safe=signals-only whitelist; full=all overrides; auto=apply overrides only if metrics improve")
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


# --------- Metric comparison: decide whether candidate is better --------- #
def _is_better(candidate: Dict, base: Dict) -> bool:
    """
    Returns True iff candidate is not worse than base on key dimensions.
    Priority:
      1) ROI higher (>= base)
      2) Profit Factor higher (>= base - tiny eps)
      3) Sharpe higher (>= base - tiny eps)
      4) Max DD not worse (<= base + small tol)
    """
    eps = 1e-6
    roi_c, roi_b = float(candidate.get("ROI", 0.0)), float(base.get("ROI", 0.0))
    pf_c,  pf_b  = float(candidate.get("profit_factor", 0.0)), float(base.get("profit_factor", 0.0))
    sh_c,  sh_b  = float(candidate.get("sharpe", 0.0)), float(base.get("sharpe", 0.0))
    dd_c,  dd_b  = float(candidate.get("max_dd_perc", 0.0)), float(base.get("max_dd_perc", 0.0))

    # Require ROI >= base (primary guard)
    if roi_c + eps < roi_b:
        return False
    # Profit factor guard (allow equal within small margin)
    if pf_c + 0.01 < pf_b:
        return False
    # Sharpe guard
    if sh_c + 0.05 < sh_b:
        return False
    # DD guard (candidate shouldn't be meaningfully worse)
    if dd_c > dd_b + 0.5:
        return False
    return True


# ---------------- MAIN ---------------- #
def main() -> None:
    args = _parse_args()
    outdir = _ensure_dirs(args.outdir)

    print(f"🧾 Fetching Zerodha OHLC: symbol={args.underlying} interval={args.interval} "
          f"{args.start} -> {args.end}")
    prices = get_zerodha_ohlc(args.underlying, args.start, args.end, args.interval)

    if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
        (outdir / "metrics.json").write_text(json.dumps({"error": "no_data"}), encoding="utf-8")
        raise SystemExit("No OHLC data returned — check Zerodha credentials or date range.")

    # ---------- Legacy runtime cfg (your high-growth baseline) ---------- #
    base_runtime_cfg = {
        "capital_rs": float(args.capital_rs),
        "order_qty": int(args.order_qty),
        "paper_trading": True,
        "live_trading": False,
    }
    profile_name = _block_to_profile(args.use_block)
    print(f"⚙️ Trade config = {{'order_qty': {base_runtime_cfg['order_qty']}, "
          f"'capital_rs': {base_runtime_cfg['capital_rs']}}}")
    print(f"🧱 Profile: {profile_name} | Mode: {args.profile_mode}")

    # Baseline plan for signals (legacy behaviour)
    compat_plan = base_runtime_cfg | {
        "backtest": {},
        "market_tz": "Asia/Kolkata"
    }

    # YAML (optional)
    cfg_yaml = _safe_load_yaml(args.config)
    base_block = cfg_yaml.get("backtest") or {}
    prof_block = cfg_yaml.get(args.use_block) or {}
    merged = _deep_merge(base_block, prof_block) if (base_block or prof_block) else {}

    # Whitelists
    SAFE_SIGNAL_KEYS = {
        "risk_perc", "atr_len", "atr_mult_sl", "atr_mult_tp",
        "allow_shorts", "trend_filter", "htf_minutes",
        # NOTE: strategy NOT in safe-set; only full/auto will try it guarded
    }
    SAFE_ENGINE_KEYS = {
        "min_hold_bars", "cooldown_bars", "trail_after_hold",
        "atr_mult_sl", "atr_mult_tp",
        # session_end/tz rarely needed; omit by default for stability
    }

    # ---------- Build BASE (legacy) signals and backtest ---------- #
    df_base = prepare_signals(prices, compat_plan)
    engine_base_cfg = {"backtest": deepcopy(base_runtime_cfg)} | deepcopy(base_runtime_cfg)
    base_summary, base_trades, base_eq = run_backtest(
        df_in=df_base,
        cfg=engine_base_cfg,
        use_block=args.use_block,
    )

    decision = {
        "mode": args.profile_mode,
        "applied": "base_only",
        "reason": "profile_mode=off or candidate not better",
        "base": base_summary,
        "candidate": None,
        "accepted_metrics": base_summary,
    }

    # ---------- Candidate: apply overrides as per mode ---------- #
    apply_candidate = args.profile_mode in {"safe", "full", "auto"} and bool(merged)

    if apply_candidate:
        if args.profile_mode == "safe":
            plan_cand = compat_plan | {k: v for k, v in merged.items() if k in SAFE_SIGNAL_KEYS}
        elif args.profile_mode == "full":
            plan_cand = compat_plan | merged
        else:  # auto: try a slightly wider set, including strategy if present
            allow_keys = set(SAFE_SIGNAL_KEYS) | {"strategy"}
            plan_cand = compat_plan | {k: v for k, v in merged.items() if k in allow_keys}

        # Signals with candidate
        df_cand = prepare_signals(prices, plan_cand)

        # Engine config with candidate (only SAFE on safe/auto; FULL on full)
        if args.profile_mode == "full":
            engine_cand_cfg = {"backtest": deepcopy(base_runtime_cfg)} | deepcopy(base_runtime_cfg)
            engine_cand_cfg["backtest"].update({k: v for k, v in merged.items()
                                                if k in SAFE_ENGINE_KEYS or k in {"session_end", "market_tz", "tz"}})
        else:
            engine_cand_cfg = {"backtest": deepcopy(base_runtime_cfg)} | deepcopy(base_runtime_cfg)
            engine_cand_cfg["backtest"].update({k: v for k, v in merged.items() if k in SAFE_ENGINE_KEYS})

        cand_summary, cand_trades, cand_eq = run_backtest(
            df_in=df_cand, cfg=engine_cand_cfg, use_block=args.use_block
        )

        decision["candidate"] = cand_summary

        # Auto mode: accept only if better; Safe/Full: accept unconditionally
        accept = (args.profile_mode in {"safe", "full"}) or _is_better(cand_summary, base_summary)
        if accept:
            # Save candidate as the final
            save_reports(outdir=outdir, summary=cand_summary, trades_df=cand_trades, equity_ser=cand_eq)
            decision["applied"] = "candidate"
            decision["reason"] = "mode=safe/full OR auto-improved"
            decision["accepted_metrics"] = cand_summary
        else:
            # Keep base as final
            save_reports(outdir=outdir, summary=base_summary, trades_df=base_trades, equity_ser=base_eq)
            decision["applied"] = "base_only"
            decision["reason"] = "auto: candidate not better; kept base"
            decision["accepted_metrics"] = base_summary
    else:
        # Profile mode off or no merged overrides: keep base
        save_reports(outdir=outdir, summary=base_summary, trades_df=base_trades, equity_ser=base_eq)

    # Write decision note
    (Path(outdir) / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"📌 Decision: {decision['applied']}  |  reason: {decision['reason']}")
    print("✅ Backtest finished & reports saved.")


if __name__ == "__main__":
    main()
