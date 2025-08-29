# core_backtest/options_legs.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

from bot.metrics import compute_metrics

# --- Black–Scholes helpers (simplified intraday) ---
from math import log, sqrt, exp
from scipy.stats import norm

def _bs_price(spot, strike, t, r, iv, call=True):
    # guard
    if t <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        intrinsic = max(spot - strike, 0.0) if call else max(strike - spot, 0.0)
        return float(intrinsic)
    d1 = (log(spot/strike) + (r + 0.5*iv*iv)*t) / (iv*sqrt(t))
    d2 = d1 - iv*sqrt(t)
    if call:
        return float(spot*norm.cdf(d1) - strike*exp(-r*t)*norm.cdf(d2))
    else:
        return float(strike*exp(-r*t)*norm.cdf(-d2) - spot*norm.cdf(-d1))

def _round_strike(x, step):
    return int(round(x / step) * step)

def _intraday_time_fraction(idx: pd.Index):
    """
    Convert a bar index (intraday) to remaining year fraction to expiry.
    For simplicity: assume same-day weekly expiry; decay during session.
    """
    n = len(idx)
    # 1 trading day ≈ 252 trading days/yr -> 1/252 year
    # Linearly decreasing within the session (more realistic intraday theta)
    base = 1.0/252.0
    # earliest bar has ~base, latest bar ~small > 0
    t = np.linspace(base, base*0.05, n)
    return t

def simulate(prices: pd.DataFrame, cfg: dict):
    """
    CE/PE leg simulation on *synthetic* premium using Black–Scholes on the fly.
    - Entries from simple MA cross on option premium (derived from spot)
    - Per-leg ATR-like stop via premium ATR proxy
    - Target = take_rr * stop
    - Trailing optional
    - Re-entry and cooldown
    Returns: (summary, trades_df, equity_series)
    """
    step   = int(cfg.get("strike_step", 100))
    iv0    = float(cfg["options"]["iv_hint"])
    r      = float(cfg["options"]["rate"])
    qty    = int(cfg.get("order_qty", 1))
    capital= float(cfg.get("capital_rs", 100000.0))

    # derive ATM strikes for each bar
    spot = prices["Close"].astype(float).values
    strikes = np.array([_round_strike(s, step) for s in spot])

    # remaining time to expiry per bar
    t_arr = _intraday_time_fraction(prices.index)

    # build CE/PE synthetic premium series
    ce = np.array([_bs_price(s, k, t, r, iv0, call=True)  for s,k,t in zip(spot, strikes, t_arr)])
    pe = np.array([_bs_price(s, k, t, r, iv0, call=False) for s,k,t in zip(spot, strikes, t_arr)])

    df = prices.copy()
    df["CE"] = ce
    df["PE"] = pe

    # basic moving average for entries on premium (per-leg engine)
    ma_len = int(cfg["entry"]["ema_fast"])
    df["CE_MA"] = pd.Series(ce).rolling(ma_len).mean()
    df["PE_MA"] = pd.Series(pe).rolling(ma_len).mean()

    # simple entry: cross above MA -> BUY that leg
    df["ce_buy"] = (df["CE"] > df["CE_MA"]) & df["CE_MA"].notna()
    df["pe_buy"] = (df["PE"] > df["PE_MA"]) & df["PE_MA"].notna()

    # premium "ATR" proxy: rolling std * k
    atr_len = int(cfg["entry"]["atr_len"])
    prem_vol_k = 1.0  # scalar -> treat as ATR-ish distance
    df["CE_ATR"] = pd.Series(ce).rolling(atr_len).std().fillna(0) * prem_vol_k
    df["PE_ATR"] = pd.Series(pe).rolling(atr_len).std().fillna(0) * prem_vol_k

    stop_k = float(cfg["risk"]["stop_atr_mult"])
    take_rr= float(cfg["risk"]["take_rr"])

    pos = None  # {"leg":"CE"/"PE","entry_px":...,"stop":...,"target":...}
    re_count = 0
    cooldown = int(cfg["reentry"]["cooldown_bars"])
    re_max   = int(cfg["reentry"]["max_per_day"])
    last_exit_i = -10**9

    eq = capital
    eq_curve = []
    trades = []

    idx = list(df.index)
    for i, ts in enumerate(idx):
        row = df.loc[ts]

        # ensure day-based reentry reset
        if i > 0:
            if pd.Timestamp(idx[i-1]).date() != pd.Timestamp(ts).date():
                re_count = 0

        # mark-to-market
        if pos:
            px = row[pos["leg"]]
            # trailing on premium: move stop closer if profit beyond k*ATR
            if cfg["trailing"]["enabled"]:
                atrp = row[f"{pos['leg']}_ATR"]
                start = cfg["trailing"]["trail_start_atr"] * atrp
                trail = cfg["trailing"]["trail_atr_mult"] * atrp
                if (px - pos["entry_px"]) >= start:
                    pos["stop"] = max(pos["stop"], px - trail)

            # exits
            if px <= pos["stop"]:
                pnl = (pos["stop"] - pos["entry_px"]) * qty
                trades[-1].update(exit_time=ts, exit=pos["stop"], reason="STOP", pnl=pnl)
                eq += pnl; pos = None; last_exit_i = i
            elif px >= pos["target"]:
                pnl = (pos["target"] - pos["entry_px"]) * qty
                trades[-1].update(exit_time=ts, exit=pos["target"], reason="TARGET", pnl=pnl)
                eq += pnl; pos = None; last_exit_i = i

        # entries (cooldown & reentry)
        if pos is None and (i - last_exit_i) >= cooldown:
            # prefer CE if both fire; else whichever fires
            if bool(row["ce_buy"]) and (re_count < re_max or re_max == 0):
                entry = row["CE"]
                atrp  = row["CE_ATR"]
                stop  = entry - stop_k * atrp
                tgt   = entry + take_rr * stop_k * atrp
                pos = {"leg":"CE","entry_px":entry,"stop":stop,"target":tgt}
                trades.append(dict(leg="CE", entry_time=ts, entry=entry))
                re_count += 1 if last_exit_i > -10**8 else 0

            elif bool(row["pe_buy"]) and (re_count < re_max or re_max == 0):
                entry = row["PE"]
                atrp  = row["PE_ATR"]
                stop  = entry - stop_k * atrp
                tgt   = entry + take_rr * stop_k * atrp
                pos = {"leg":"PE","entry_px":entry,"stop":stop,"target":tgt}
                trades.append(dict(leg="PE", entry_time=ts, entry=entry))
                re_count += 1 if last_exit_i > -10**8 else 0

        eq_curve.append(eq)

    trades_df = pd.DataFrame(trades)
    equity = pd.Series(eq_curve, index=df.index, name="equity")
    summary = compute_metrics(trades_df, equity, capital)
    return summary, trades_df, equity
