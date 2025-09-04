# bot/backtest.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict

import pandas as pd
import numpy as np

from bot.strategy import prepare_signals


# ------------------------
# Session helper (FIX for Pandas Timestamp 'year' error)
# ------------------------
def _parse_hhmm(hhmm: str) -> pd.Timestamp:
    # robust: accepts "9:20", "09:20", "09:20:00"
    hhmm = (hhmm or "").strip()
    # stick an arbitrary date to make a full timestamp
    return pd.to_datetime(f"2000-01-01 {hhmm}", format="%Y-%m-%d %H:%M", errors="coerce")

def _within_session(ts: pd.Timestamp, sess_s: str, sess_e: str) -> bool:
    if pd.isna(ts):
        return False
    sh = _parse_hhmm(sess_s)
    eh = _parse_hhmm(sess_e)
    if pd.isna(sh) or pd.isna(eh):
        return True  # if misconfigured, don't block
    t = pd.to_datetime(f"2000-01-01 {ts.time()}")
    return (t.time() >= sh.time()) and (t.time() <= eh.time())


# ------------------------
# Very small trade engine (long-only)
# ------------------------
@dataclass
class BTState:
    in_pos: bool = False
    entry_px: float = 0.0

def _simulate(df: pd.DataFrame, cfg: dict, profile: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Very small long-only backtest using prepared entry/exit hints.
    Produces trades dataframe and equity curve (by closed trades).
    """
    st = BTState()
    trades = []
    equity = []
    cash = float(cfg.get("capital_rs", 100000.0))
    qty = int(cfg.get("order_qty", 1))

    sess_s = (cfg.get("backtest") or {}).get("session_start", "09:20")
    sess_e = (cfg.get("backtest") or {}).get("session_end", "15:20")

    for ts, row in df.iterrows():
        if not _within_session(ts, sess_s, sess_e):
            continue

        px = float(row["Close"])
        # enter
        if (not st.in_pos) and bool(row.get("enter_long", False)):
            st.in_pos = True
            st.entry_px = px

        # exit
        if st.in_pos and bool(row.get("exit_long_hint", False)):
            pnl = (px - st.entry_px) * qty
            trades.append({"ts": ts, "entry": st.entry_px, "exit": px, "pnl": pnl})
            cash += pnl
            equity.append(cash)
            st = BTState()  # reset

    # if open trade at end, mark-to-close
    if st.in_pos:
        px = float(df["Close"].iloc[-1])
        pnl = (px - st.entry_px) * qty
        trades.append({"ts": df.index[-1], "entry": st.entry_px, "exit": px, "pnl": pnl})
        cash += pnl
        equity.append(cash)
        st = BTState()

    trades_df = pd.DataFrame(trades)
    equity_ser = pd.Series(equity, name="equity")

    return trades_df, equity_ser


def _summarize(df_prices: pd.DataFrame,
               trades_df: pd.DataFrame,
               equity_ser: pd.Series,
               cfg: dict,
               profile: str) -> Dict:
    """
    Build a summary dict (keys used by reports).
    """
    trades = len(trades_df)
    wins = int((trades_df["pnl"] > 0).sum()) if trades else 0
    win_rate = (wins / trades * 100.0) if trades else 0.0
    roi = 0.0
    pf = 0.0
    rr = 0.0
    sharpe = -0.00
    max_dd_pct = 0.0
    time_dd_bars = len(df_prices)

    if trades:
        gross_p = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
        gross_l = -trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()
        pf = (gross_p / gross_l) if gross_l > 0 else 0.0
        rr = (abs(trades_df["pnl"].mean()) / (trades_df["pnl"].abs().mean() + 1e-9)) if trades else 0.0

        start_cap = float(cfg.get("capital_rs", 100000.0))
        end_cap = equity_ser.iloc[-1] if len(equity_ser) else start_cap
        roi = (end_cap - start_cap) / start_cap * 100.0

        # rough drawdown from equity
        if len(equity_ser):
            roll_max = equity_ser.cummax()
            dd = (equity_ser - roll_max)
            max_dd_abs = dd.min() if len(dd) else 0.0
            max_dd_pct = (max_dd_abs / start_cap) * 100.0

    summary = dict(
        underlying=cfg.get("underlying", "NIFTY"),
        interval="1m",
        period_start=str(df_prices.index.min())[:10] if len(df_prices) else "",
        period_end=str(df_prices.index.max())[:10] if len(df_prices) else "",
        trades=trades,
        win_rate_pct=round(win_rate, 2),
        roi_pct=round(roi, 2),
        pf=round(pf, 2),
        rr=round(rr, 2),
        sharpe=round(sharpe, 2),
        max_dd_pct=round(max_dd_pct, 2),
        time_dd_bars=int(time_dd_bars),
        bars=int(len(df_prices)),
        atr_bars=0,
        setups_long=int(df_prices.get("enter_long", pd.Series(dtype=bool)).sum() if len(df_prices) else 0),
        setups_short=0,
        profile=profile,
    )
    return summary


# ------------------------
# Public API
# ------------------------
def run_backtest(prices: pd.DataFrame,
                 cfg: dict,
                 profile: str = "loose",
                 use_block: str = "") -> tuple[dict, pd.DataFrame, pd.Series]:
    """
    Main entry used by tools/run_backtest.py
    Returns: (summary_dict, trades_df, equity_series)
    """
    if prices is None or len(prices) == 0:
        # empty inputs -> empty outputs
        empty = pd.DataFrame(), pd.Series(dtype=float)
        return _summarize(pd.DataFrame(), *empty, cfg, profile), *empty

    # 1) build indicators & signals
    df = prepare_signals(prices, cfg, profile=profile)

    # 2) simulate
    trades_df, equity_ser = _simulate(df, cfg, profile)

    # 3) summarize
    summary = _summarize(df, trades_df, equity_ser, cfg, profile)

    return summary, trades_df, equity_ser
