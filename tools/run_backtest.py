# tools/run_backtest.py
from __future__ import annotations

import argparse, json
from copy import deepcopy
from pathlib import Path
from typing import Dict
import pandas as pd
import datetime as dt, pytz

# optional: YAML for config
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from bot.data_io import get_zerodha_ohlc
from bot.strategy import prepare_signals
from bot.backtest import run_backtest, save_reports

# --- IST tz ---
IST = pytz.timezone("Asia/Kolkata")

# ---------------- LIVE-MODE helpers ---------------- #
def _pick_datetime_col(df: pd.DataFrame):
    """auto detect best datetime column"""
    candidates = ["entry_time", "entry_at", "entry", "timestamp", "time", "date"]
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in low: return low[c]
    return None

def _to_ist_date(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            return s.dt.tz_convert(IST).dt.date
    except Exception:
        pass
    return s.dt.date

def _filter_today_ist(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return trades_df
    col = _pick_datetime_col(trades_df)
    if not col: return trades_df
    today = dt.datetime.now(IST).date()
    d = _to_ist_date(trades_df[col])
    out = trades_df.loc[d == today].copy()
    print(f"🔎 live-filter: column={col}, kept={len(out)}/{len(trades_df)} for {today}")
    return out

def _save_final_reports(outdir: Path, summary: Dict, trades_df: pd.DataFrame,
                        equity_ser: pd.Series, live_mode: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if live_mode:
        (outdir / "trades_full_window.csv").write_text(trades_df.to_csv(index=False), encoding="utf-8")
        trades_today = _filter_today_ist(trades_df)
        if trades_today is None or trades_today.empty:
            print("ℹ️ no trades today; writing empty trades.csv")
        save_reports(outdir=outdir, summary=summary, trades_df=trades_today, equity_ser=equity_ser)
    else:
        save_reports(outdir=outdir, summary=summary, trades_df=trades_df, equity_ser=equity_ser)

# ---------------- CLI ---------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run backtest (with live-mode support)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--underlying", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--interval", default="1m")
    p.add_argument("--capital_rs", type=float, default=100000.0)
    p.add_argument("--order_qty", type=int, default=1)
    p.add_argument("--outdir", required=True)
    p.add_argument("--use_block", default="backtest_loose",
                   choices=["backtest_loose", "backtest_medium", "backtest_strict"])
    p.add_argument("--profile_mode", default="auto",
                   choices=["off", "safe", "full", "auto"])
    p.add_argument("--live_mode", action="store_true",
                   help="save only today's (IST) trades")
    p.add_argument("--warmup_days", type=int, default=5,
                   help="history fetched for warm-up")
    return p.parse_args()

def _ensure_dirs(outdir: str) -> Path:
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    return p

def _safe_load_yaml(path: str) -> dict:
    if yaml is None: return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception: return {}

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

def _is_better(candidate: Dict, base: Dict) -> bool:
    eps = 1e-6
    roi_c, roi_b = float(candidate.get("ROI", 0.0)), float(base.get("ROI", 0.0))
    pf_c,  pf_b  = float(candidate.get("profit_factor", 0.0)), float(base.get("profit_factor", 0.0))
    sh_c,  sh_b  = float(candidate.get("sharpe", 0.0)), float(base.get("sharpe", 0.0))
    dd_c,  dd_b  = float(candidate.get("max_dd_perc", 0.0)), float(base.get("max_dd_perc", 0.0))
    if roi_c + eps < roi_b: return False
    if pf_c + 0.01 < pf_b:  return False
    if sh_c + 0.05 < sh_b:  return False
    if dd_c > dd_b + 0.5:   return False
    return True

# ---------------- MAIN ---------------- #
def main() -> None:
    args = _parse_args()
    outdir = _ensure_dirs(args.outdir)

    print(f"🧾 Fetching Zerodha OHLC: {args.underlying} {args.interval} {args.start} → {args.end}")
    prices = get_zerodha_ohlc(args.underlying, args.start, args.end, args.interval)
    if prices is None or (isinstance(prices, pd.DataFrame) and prices.empty):
        (outdir / "metrics.json").write_text(json.dumps({"error": "no_data"}), encoding="utf-8")
        raise SystemExit("No OHLC data returned")

    base_runtime_cfg = {
        "capital_rs": float(args.capital_rs),
        "order_qty": int(args.order_qty),
        "paper_trading": True,
        "live_trading": False,
    }
    profile_name = _block_to_profile(args.use_block)
    print(f"⚙️ order_qty={base_runtime_cfg['order_qty']} capital={base_runtime_cfg['capital_rs']}")
    print(f"🧱 Profile={profile_name} | Mode={args.profile_mode}")

    compat_plan = base_runtime_cfg | {"backtest": {}, "market_tz": "Asia/Kolkata"}

    cfg_yaml = _safe_load_yaml(args.config)
    base_block = cfg_yaml.get("backtest") or {}
    prof_block = cfg_yaml.get(args.use_block) or {}
    merged = _deep_merge(base_block, prof_block) if (base_block or prof_block) else {}

    SAFE_SIGNAL_KEYS = {"risk_perc","atr_len","atr_mult_sl","atr_mult_tp","allow_shorts","trend_filter","htf_minutes"}
    SAFE_ENGINE_KEYS = {"min_hold_bars","cooldown_bars","trail_after_hold","atr_mult_sl","atr_mult_tp"}

    df_base = prepare_signals(prices, compat_plan)
    engine_base_cfg = {"backtest": deepcopy(base_runtime_cfg)} | deepcopy(base_runtime_cfg)
    base_summary, base_trades, base_eq = run_backtest(df_in=df_base, cfg=engine_base_cfg, use_block=args.use_block)

    decision = {"mode": args.profile_mode, "applied": "base_only",
                "reason": "profile_mode=off or candidate not better",
                "base": base_summary, "candidate": None,
                "accepted_metrics": base_summary}

    apply_candidate = args.profile_mode in {"safe","full","auto"} and bool(merged)
    if apply_candidate:
        if args.profile_mode == "safe":
            plan_cand = compat_plan | {k: v for k, v in merged.items() if k in SAFE_SIGNAL_KEYS}
        elif args.profile_mode == "full":
            plan_cand = compat_plan | merged
        else:
            allow_keys = set(SAFE_SIGNAL_KEYS)|{"strategy"}
            plan_cand = compat_plan | {k: v for k,v in merged.items() if k in allow_keys}
        df_cand = prepare_signals(prices, plan_cand)
        engine_cand_cfg = {"backtest": deepcopy(base_runtime_cfg)} | deepcopy(base_runtime_cfg)
        engine_cand_cfg["backtest"].update({k:v for k,v in merged.items() if k in SAFE_ENGINE_KEYS})
        cand_summary, cand_trades, cand_eq = run_backtest(df_in=df_cand, cfg=engine_cand_cfg, use_block=args.use_block)
        decision["candidate"] = cand_summary
        accept = (args.profile_mode in {"safe","full"}) or _is_better(cand_summary, base_summary)
        if accept:
            _save_final_reports(outdir, cand_summary, cand_trades, cand_eq, args.live_mode)
            decision.update({"applied":"candidate","reason":"improved","accepted_metrics":cand_summary})
        else:
            _save_final_reports(outdir, base_summary, base_trades, base_eq, args.live_mode)
    else:
        _save_final_reports(outdir, base_summary, base_trades, base_eq, args.live_mode)

    (Path(outdir)/"decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"📌 Decision={decision['applied']} reason={decision['reason']}")
    print("✅ Finished & reports saved.")

if __name__=="__main__":
    main()
