# bot/backtest.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import time as dtime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Charts headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# If signals not already prepared, we can compute a basic set
try:
    from bot.strategy import prepare_signals
except Exception:
    prepare_signals = None   # optional fallback


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_hhmm(s: str) -> dtime:
    h, m = s.split(":")
    return dtime(int(h), int(m))

def _within_session(ts: pd.Timestamp, sess_start: str, sess_end: str) -> bool:
    """
    Compare only time-of-day (avoid Pandas 2.x Timestamp(hour=...) issue).
    """
    tt = ts.tz_convert(None).time() if getattr(ts, "tzinfo", None) else ts.time()
    s = _parse_hhmm(sess_start)
    e = _parse_hhmm(sess_end)
    return (tt >= s) and (tt <= e)


def _safe_int_qty(val, default_qty: int) -> int:
    """
    Convert any numeric-ish value to a positive int.
    Falls back to default_qty if NaN/inf/<=0/invalid.
    """
    try:
        qf = float(val)
    except Exception:
        return int(default_qty)

    if not np.isfinite(qf) or qf <= 0:
        return int(default_qty)

    return int(max(1, round(qf)))


@dataclass
class Position:
    side: str = "long"      # only long for now
    entry_px: float = 0.0
    sl_px: float = 0.0
    tp_px: float = 0.0
    qty: int = 0
    entry_ts: pd.Timestamp | None = None


# ---------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------

def _ensure_signals(df: pd.DataFrame, cfg: dict, profile_name: str) -> pd.DataFrame:
    """
    If dataframe doesn't contain required columns, try to create them via
    bot.strategy.prepare_signals (if available). Otherwise add safe defaults.
    """
    need = {"enter_long", "atr"}
    if need.issubset(set(df.columns)):
        return df

    if prepare_signals is not None:
        try:
            # derive profile from use_block like "backtest_loose" -> "loose"
            prof = (profile_name or "backtest_loose").replace("backtest_", "")
            return prepare_signals(df, cfg, profile=prof)
        except Exception:
            pass  # fall back to safe defaults

    # Safe minimal defaults – will result in zero trades but no crash
    out = df.copy()
    out["atr"] = out["Close"].rolling(14).apply(
        lambda _: np.nan, raw=True
    )  # keep NaNs, we guard later
    out["enter_long"] = False
    return out


def run_backtest(
    prices: pd.DataFrame,
    cfg: dict,
    use_block: str = "backtest_loose"
):
    """
    Long-only backtest with simple SL/TP in ATR multiples.
    Returns: (summary: dict, trades_df: DataFrame, equity_ser: Series)
    """
    df = prices.copy().sort_index()
    df = _ensure_signals(df, cfg, use_block)

    # ---- parameters (backward-compatible reads) ----
    bcfg = (cfg.get("backtest") or {})
    filters = (bcfg.get("filters") or {})

    sess_s   = filters.get("session_start", bcfg.get("session_start", "09:20"))
    sess_e   = filters.get("session_end",   bcfg.get("session_end",   "15:20"))
    stop_mult = float((bcfg.get("exits") or {}).get("stop_atr_mult", cfg.get("stop_atr_mult", 1.0)))
    take_mult = float((bcfg.get("exits") or {}).get("take_atr_mult", cfg.get("take_atr_mult", 1.3)))

    capital      = float(cfg.get("capital_rs", 100_000.0))
    default_qty  = int(cfg.get("order_qty", 1))

    # Optional position sizing by ATR (pos_size column).
    # If caller already prepared pos_size, we just sanitize it.
    # Else create conservative sizing (1 lot) unless atr is present & finite.
    if "pos_size" not in df.columns:
        # Example: size inversely proportional to ATR (very conservative cap)
        # Avoid NaNs/inf
        atr = pd.to_numeric(df.get("atr", pd.Series([np.nan]*len(df), index=df.index)), errors="coerce")
        px  = pd.to_numeric(df["Close"], errors="coerce")
        # risk per trade (₹) ~ 0.25% of capital, size = risk / (ATR * k)
        risk_rs = 0.0025 * capital
        k = max(1.0, stop_mult)  # stop distance ~ ATR*stop_mult
        raw = risk_rs / (atr * k)
        df["pos_size"] = raw.clip(lower=1, upper=default_qty*10)  # cap growth
    # sanitize to integer qty safely (no astype that can raise)
    df["pos_size_sanitized"] = [
        _safe_int_qty(v, default_qty) for v in df["pos_size"].tolist()
    ]

    # ---- run state ----
    pos: Position | None = None
    cash = capital
    eq_curve: list[tuple[pd.Timestamp, float]] = []
    trades: list[dict] = []

    for ts, row in df.iterrows():
        # session window check
        if not _within_session(ts, sess_s, sess_e):
            if pos is not None:
                # square-off
                pnl = (float(row["Close"]) - pos.entry_px) * pos.qty
                cash += pnl
                trades.append(dict(
                    entry_ts=pos.entry_ts, exit_ts=ts, side="long",
                    entry=pos.entry_px, exit=float(row["Close"]),
                    qty=int(pos.qty), pnl=float(pnl), reason="EOD"
                ))
                pos = None
            eq_curve.append((ts, cash))
            continue

        # entry
        if pos is None and bool(row.get("enter_long", False)):
            entry = float(row["Close"])
            atr   = float(row.get("atr", 0.0)) if np.isfinite(row.get("atr", 0.0)) else 0.0
            sl    = entry - stop_mult * atr
            tp    = entry + take_mult * atr
            qty   = _safe_int_qty(row.get("pos_size_sanitized", default_qty), default_qty)

            pos = Position(
                side="long", entry_px=entry, sl_px=sl, tp_px=tp,
                qty=qty, entry_ts=ts
            )

        # manage open
        if pos is not None:
            low  = float(row.get("Low",  row["Close"]))
            high = float(row.get("High", row["Close"]))
            exit_px = None
            reason  = None

            if low <= pos.sl_px:
                exit_px = pos.sl_px; reason = "SL"
            elif high >= pos.tp_px:
                exit_px = pos.tp_px; reason = "TP"
            elif bool(row.get("exit_long_hint", False)):
                # optional soft exit: close at market
                exit_px = float(row["Close"]); reason = "HINT"

            if exit_px is not None:
                pnl = (exit_px - pos.entry_px) * pos.qty
                cash += pnl
                trades.append(dict(
                    entry_ts=pos.entry_ts, exit_ts=ts, side="long",
                    entry=pos.entry_px, exit=float(exit_px),
                    qty=int(pos.qty), pnl=float(pnl), reason=reason
                ))
                pos = None

        eq_curve.append((ts, cash))

    # final square-off if still open
    if pos is not None:
        last_ts = df.index[-1]
        last_px = float(df.iloc[-1]["Close"])
        pnl = (last_px - pos.entry_px) * pos.qty
        cash += pnl
        trades.append(dict(
            entry_ts=pos.entry_ts, exit_ts=last_ts, side="long",
            entry=pos.entry_px, exit=last_px,
            qty=int(pos.qty), pnl=float(pnl), reason="EOD"
        ))
        pos = None
        eq_curve[-1] = (last_ts, cash)

    equity_ser = pd.Series({ts: val for ts, val in eq_curve}).sort_index()

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    base = capital
    ret = (equity_ser.iloc[-1] - base) / base if len(equity_ser) else 0.0

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        # ensure safe numeric dtypes without crashing on NaN/inf
        for col in ["qty", "entry", "exit", "pnl"]:
            trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")
        trades_df["qty"] = trades_df["qty"].fillna(0).round().astype(int)

    if trades_df.empty:
        win = 0.0; rr = 0.0; pf = 0.0
    else:
        win = float((trades_df["pnl"] > 0).mean() * 100.0)
        has_pos = (trades_df["pnl"] > 0).any()
        has_neg = (trades_df["pnl"] < 0).any()
        rr = (
            trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
            / abs(trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean())
        ) if (has_pos and has_neg) else 0.0
        pf = (
            trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
            / abs(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())
        ) if (has_pos and has_neg) else 0.0

    # drawdown in ₹ terms then % of base
    roll_max = equity_ser.cummax() if len(equity_ser) else equity_ser
    dd = equity_ser - roll_max if len(equity_ser) else equity_ser
    max_dd = float((dd.min() / base) * 100) if len(dd) else 0.0
    tdd_bars = int((dd == dd.min()).sum()) if len(dd) else 0

    summary = dict(
        n_trades=int(len(trades_df)),
        win_rate=round(win, 2),
        roi_pct=round(ret * 100, 2),
        profit_factor=round(float(pf), 2),
        rr=round(float(rr), 2),
        sharpe_ratio=0.0,                 # placeholder
        max_dd_pct=round(max_dd, 2),
        time_dd_bars=tdd_bars,
        n_bars=int(len(df)),
        atr_bars=int(pd.to_numeric(df.get("atr", pd.Series([])), errors="coerce").gt(0).sum()),
        setups_long=int(pd.to_numeric(df.get("enter_long", pd.Series([])), errors="coerce").fillna(0).astype(bool).sum()),
        setups_short=0,
    )

    return summary, trades_df, equity_ser


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def _plot_equity(equity: pd.Series, out: Path):
    if equity is None or equity.empty:
        return
    plt.figure()
    equity.plot()
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity (₹)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def _plot_drawdown(equity: pd.Series, out: Path):
    if equity is None or equity.empty:
        return
    roll_max = equity.cummax()
    dd = equity - roll_max
    plt.figure()
    dd.plot()
    plt.title("Drawdown (₹)")
    plt.xlabel("Trade #")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def save_reports(outdir, summary: Dict, trades_df: pd.DataFrame, equity_ser: pd.Series):
    """
    Writes:
      - metrics.json
      - trades.csv (if any)
      - equity.csv (if any)
      - equity_curve.png / drawdown.png (if any)
      - report.md
    """
    out_path = Path(outdir) if not isinstance(outdir, Path) else outdir
    out_path.mkdir(parents=True, exist_ok=True)

    # JSON metrics
    (out_path / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # CSVs
    if equity_ser is not None and not equity_ser.empty:
        equity_ser.to_csv(out_path / "equity.csv", header=["equity"], index_label="ts")

    if trades_df is not None and not trades_df.empty:
        # ensure integer column for qty without crashing
        if "qty" in trades_df.columns:
            trades_df["qty"] = pd.to_numeric(trades_df["qty"], errors="coerce").fillna(0).round().astype(int)
        trades_df.to_csv(out_path / "trades.csv", index=False)

    # Charts
    _plot_equity(equity_ser, out_path / "equity_curve.png")
    _plot_drawdown(equity_ser, out_path / "drawdown.png")

    # Simple markdown
    lines = [
        "# Backtest Report",
        "",
        f"**Trades:** {summary.get('n_trades', 0)}",
        f"**Win-rate:** {summary.get('win_rate', 0)}%",
        f"**ROI:** {summary.get('roi_pct', 0)}%",
        f"**PF:** {summary.get('profit_factor', 0)}",
        f"**R:R:** {summary.get('rr', 0)}",
        f"**Max DD:** {summary.get('max_dd_pct', 0)}%",
        "",
        "Charts: `equity_curve.png`, `drawdown.png`",
    ]
    (out_path / "report.md").write_text("\n".join(lines), encoding="utf-8")
