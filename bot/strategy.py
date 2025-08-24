# bot/strategy.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# =========================
# Daily swing helpers you already had (keep as-is if you want)
# =========================

def build_signals(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Your existing daily signals builder (ema/rsi/adx based).
    Kept for compatibility with runner.py.
    """
    d = df.copy()
    # expect columns: Date, Open, High, Low, Close
    # label simply from some rule you already used
    d["ema_fast"] = d["Close"].ewm(span=cfg.get("ema_fast", 21), adjust=False).mean()
    d["ema_slow"] = d["Close"].ewm(span=cfg.get("ema_slow", 50), adjust=False).mean()
    d["label"] = np.where(d["ema_fast"] > d["ema_slow"], "BUY",
                   np.where(d["ema_fast"] < d["ema_slow"], "SELL", "HOLD"))
    return d

# =========================
# Intraday Options — ATM Short Straddle (Paper)
# =========================

@dataclass
class StraddleParams:
    lots: int
    lot_size: int
    leg_sl_pct: float
    combined_target_pct: float
    squareoff_ts: pd.Timestamp  # IST
    entry_ts: pd.Timestamp      # IST

def _ensure_tz_ist(ts: pd.Timestamp, tz: str = "Asia/Kolkata") -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)

def _align_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expect a 'Date' column (tz-aware). Index by Date and sort.
    """
    d = df.copy()
    if "Date" not in d.columns:
        d = d.rename(columns={"date": "Date"})
    d["Date"] = pd.to_datetime(d["Date"], utc=False, infer_datetime_format=True)
    # if timezone missing, DON'T double-localize; just localize once
    if d["Date"].dt.tz is None:
        d["Date"] = d["Date"].dt.tz_localize("Asia/Kolkata")
    else:
        d["Date"] = d["Date"].dt.tz_convert("Asia/Kolkata")
    d = d.sort_values("Date").set_index("Date")
    return d

def simulate_short_straddle_intraday(df_ce: pd.DataFrame,
                                     df_pe: pd.DataFrame,
                                     params: StraddleParams) -> Dict:
    """
    Very simple intraday paper backtest for a single day:
      - Entry: short CE & short PE at entry_ts (use bar close at/after that time)
      - SL per leg: leg_sl_pct of entry premium (on each leg independently)
      - Combined target: when (CE+PE) has fallen by combined_target_pct
      - Square off: at squareoff_ts if still open
    Assumes both DataFrames are 5m candles of the same date, tz-aware IST.
    """
    ce = _align_by_time(df_ce)[["Open", "High", "Low", "Close"]].copy()
    pe = _align_by_time(df_pe)[["Open", "High", "Low", "Close"]].copy()

    # pick the first bar ON/AFTER entry_ts
    ce_entry_bar = ce.loc[ce.index >= params.entry_ts]
    pe_entry_bar = pe.loc[pe.index >= params.entry_ts]
    if ce_entry_bar.empty or pe_entry_bar.empty:
        return {"status": "no_entry_bar", "pnl_rs": 0.0}

    t0 = max(ce_entry_bar.index[0], pe_entry_bar.index[0])
    ce_entry = float(ce.loc[t0, "Close"])
    pe_entry = float(pe.loc[t0, "Close"])
    qty = params.lots * params.lot_size

    # We're SHORT, so entry credit:
    entry_credit = (ce_entry + pe_entry) * qty

    # SL levels (per leg)
    ce_sl = ce_entry * (1 + params.leg_sl_pct / 100.0)
    pe_sl = pe_entry * (1 + params.leg_sl_pct / 100.0)

    combined_target = (ce_entry + pe_entry) * (1 - params.combined_target_pct / 100.0)

    ce_active, pe_active = True, True
    exit_ts = None
    exit_debit_ce = 0.0
    exit_debit_pe = 0.0

    # Iterate bars after entry until squareoff
    walk = ce.join(pe, lsuffix="_ce", rsuffix="_pe", how="outer").sort_index()
    walk = walk.loc[(walk.index >= t0) & (walk.index <= params.squareoff_ts)]

    for ts, row in walk.iterrows():
        # current prices (use Close; could refine with High/Low for SL intrabar)
        p_ce = float(row.get("Close_ce", np.nan))
        p_pe = float(row.get("Close_pe", np.nan))
        if np.isnan(p_ce) or np.isnan(p_pe):
            continue

        # Check per-leg SL
        if ce_active and p_ce >= ce_sl:
            ce_active = False
            exit_debit_ce = p_ce * qty  # buy back CE
        if pe_active and p_pe >= pe_sl:
            pe_active = False
            exit_debit_pe = p_pe * qty  # buy back PE

        # If both legs still open, check combined target
        if ce_active and pe_active:
            if (p_ce + p_pe) <= combined_target:
                ce_active = pe_active = False
                exit_debit_ce = p_ce * qty
                exit_debit_pe = p_pe * qty
                exit_ts = ts
                break

        # If both legs closed by SLs independently, we can stop
        if not ce_active and not pe_active:
            exit_ts = ts
            break

    # If still open at squareoff:
    if ce_active:
        last_ce = float(ce.loc[:params.squareoff_ts]["Close"].iloc[-1])
        exit_debit_ce = last_ce * qty
    if pe_active:
        last_pe = float(pe.loc[:params.squareoff_ts]["Close"].iloc[-1])
        exit_debit_pe = last_pe * qty
    if exit_ts is None:
        exit_ts = params.squareoff_ts

    pnl_rs = entry_credit - (exit_debit_ce + exit_debit_pe)

    return {
        "status": "ok",
        "entry_ts": str(t0),
        "exit_ts": str(exit_ts),
        "entry_ce": ce_entry,
        "entry_pe": pe_entry,
        "exit_ce": exit_debit_ce / qty,
        "exit_pe": exit_debit_pe / qty,
        "lots": params.lots,
        "lot_size": params.lot_size,
        "pnl_rs": round(pnl_rs, 2),
        "entry_credit": round(entry_credit, 2),
        "exit_debit": round(exit_debit_ce + exit_debit_pe, 2),
    }
