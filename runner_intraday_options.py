# runner_intraday_options.py
from __future__ import annotations
import os, sys, json, argparse
import pandas as pd

# Local imports
from bot.config import load_config, debug_fingerprint
from bot.data_io import prices
from bot.backtest import run_backtest, save_reports
from bot.utils import ensure_dir, save_json, send_telegram

"""
Intraday options (index-level) paper runner

• Loads config.yaml
• Pulls index candles from Zerodha (BANKNIFTY/NIFTY)
• Generates signals via bot/strategy.py (config-driven)
• Simulates entries with ATR SL/Target + trailing & re-entries
• Saves reports (trades.csv, equity.csv, metrics.json, charts, report.md)
• (Optional) Telegram summary handled by tools/telegram_notify.py workflow step
"""

def build_cfg(base: dict) -> dict:
    """Merge safe defaults that increase intraday trade frequency."""
    cfg = dict(base or {})
    io = cfg.get("intraday_options") or {}
    cfg["intraday_options"] = io

    # ---------- Intraday source ----------
    io.setdefault("underlying", "BANKNIFTY")
    io.setdefault("index_token", 260105)   # BANKNIFTY
    io.setdefault("timeframe", "5minute")
    io.setdefault("strike_step", 100)

    # ---------- Entry engine (loosened a bit) ----------
    cfg.setdefault("ema_fast", 21)
    cfg.setdefault("ema_slow", 50)
    cfg.setdefault("rsi_len", 14)
    cfg.setdefault("adx_len", 14)
    cfg.setdefault("atr_len", 14)

    # Trading mode for signal engine
    cfg.setdefault("signal_mode", "balanced")    # strict | balanced | aggressive
    cfg.setdefault("rsi_buy", 52)                # long trigger
    cfg.setdefault("rsi_sell", 48)               # short trigger
    cfg.setdefault("ema_poke_pct", 0.0005)       # 0.05% above/below EMA fast
    cfg.setdefault("adx_min_bal", 10)            # relaxed ADX for balanced
    cfg.setdefault("adx_min_strict", 18)

    # ---------- Risk / exits ----------
    cfg.setdefault("stop_atr_mult", 1.2)         # tighter so trades finish intraday
    cfg.setdefault("take_atr_mult", 1.6)
    cfg.setdefault("trailing_enabled", True)
    cfg.setdefault("trail_start_atr", 0.6)       # start trailing earlier
    cfg.setdefault("trail_atr_mult", 1.0)

    # ---------- Re-entries ----------
    cfg.setdefault("reentry_max", 3)             # up to 3 fresh attempts same day
    cfg.setdefault("reentry_cooldown", 2)        # bars to wait after exit

    # ---------- Capital / sizing ----------
    cfg.setdefault("capital_rs", 100000.0)
    cfg.setdefault("order_qty", 1)

    # ---------- Output ----------
    cfg.setdefault("out_dir", "reports")

    # ---------- Zerodha toggles ----------
    cfg.setdefault("zerodha_enabled", True)
    # (API secrets are injected via env in Actions)

    return cfg


def fetch_index_prices(cfg: dict, period: str = "30d") -> pd.DataFrame:
    """Pull index candles from Zerodha."""
    io = cfg["intraday_options"]
    token = int(io["index_token"])
    interval = io.get("timeframe", "5minute")
    df = prices(
        symbol=io.get("underlying", "BANKNIFTY"),
        period=period,
        interval=interval,
        zerodha_enabled=True,
        zerodha_instrument_token=token,
    )
    need_cols = {"Open","High","Low","Close"}
    if not need_cols.issubset(set(df.columns)):
        raise RuntimeError("Downloaded candles missing OHLC columns.")
    return df


def write_latest_summary(out_dir: str, meta: dict):
    ensure_dir(out_dir)
    save_json(meta, os.path.join(out_dir, "latest.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default=os.getenv("INTRADAY_PERIOD", "30d"),
                    help="Zerodha history lookback like 7d/30d/6mo/1y")
    ap.add_argument("--mode", default=os.getenv("SIGNAL_MODE", ""),
                    help="strict|balanced|aggressive (overrides config)")
    ap.add_argument("--out_dir", default="", help="override output dir")
    args = ap.parse_args()

    # Load + merge defaults
    cfg = build_cfg(load_config("config.yaml"))
    if args.mode.strip():
        cfg["signal_mode"] = args.mode.strip().lower()
    if args.out_dir.strip():
        cfg["out_dir"] = args.out_dir.strip()

    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    print("🔧 Config fingerprint:", debug_fingerprint(cfg))
    print("⚙️  Signal mode:", cfg.get("signal_mode"))

    # 1) Data
    print("⬇️  Downloading index candles…")
    df = fetch_index_prices(cfg, period=args.period)
    print(f"✅ Got {len(df)} bars.")

    # 2) Backtest on index-level
    print("🏃 Running intraday backtest (index-level)…")
    summary, trades, equity = run_backtest(df, cfg)

    # 3) Save artifacts
    print("💾 Saving reports…")
    save_reports(out_dir, summary, trades, equity)

    # 4) Write latest.json (used by Telegram step)
    write_latest_summary(out_dir, {
        "runner": "intraday_index",
        "signal_mode": cfg.get("signal_mode"),
        "period": args.period,
        "n_rows": int(len(df)),
    })

    # 5) (Optional) print quick line for Actions log
    print("📊 Summary:", json.dumps(summary, indent=2))

    # Optional direct Telegram (usually workflow calls tools/telegram_notify.py)
    if os.getenv("SEND_TG_NOW", "").lower() == "true":
        msg = (
            "📈 Intraday Summary\n"
            f"• Trades: {summary.get('n_trades',0)}\n"
            f"• Win-rate: {summary.get('win_rate',0)}%\n"
            f"• ROI: {summary.get('roi_pct',0)}%\n"
            f"• Profit Factor: {summary.get('profit_factor',0)}\n"
            f"• R:R: {summary.get('rr',0)}\n"
            f"• Max DD: {summary.get('max_dd_pct',0)}%\n"
            f"• Time DD (bars): {summary.get('time_dd_bars',0)}\n"
            f"• Sharpe: {summary.get('sharpe_ratio',0)}"
        )
        send_telegram(msg, cfg.get("TELEGRAM_BOT_TOKEN"), cfg.get("TELEGRAM_CHAT_ID"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Make sure CI still uploads artifacts even if something fails
        print("❌ Runner error:", repr(e))
        # Write minimal files so telegram step shows a message
        out_dir = "reports"
        ensure_dir(out_dir)
        save_json({"error": str(e)}, os.path.join(out_dir, "latest.json"))
        save_json({
            "ts": pd.Timestamp.utcnow().isoformat() + "Z",
            "n_trades": 0,
            "win_rate": 0.0,
            "roi_pct": 0.0,
            "profit_factor": 0.0,
            "rr": 0.0,
            "max_dd_pct": 0.0,
            "time_dd_bars": 0,
            "sharpe_ratio": 0.0,
            "note": "Runner failed; see logs.",
        }, os.path.join(out_dir, "metrics.json"))
        sys.exit(1)
