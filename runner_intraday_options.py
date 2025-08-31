#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intraday Options (paper) runner – self-contained imports
- No dependency on utils.telegram / broker.* / strategies.*
- Has a tiny RSI strategy + Telegram sender built-in
- Uses your existing data_io.get_index_candles if present, else a safe fallback
- Writes reports/ {metrics.json, latest.json, latest_signals.csv} like before
"""

from __future__ import annotations
import os, sys, io, json, math, time, pathlib, datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

# -----------------------------
# Config helpers
# -----------------------------
def load_yaml_cfg(path="config.yaml") -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -----------------------------
# Light-weight Telegram (no external import)
# -----------------------------
def tg_send_text(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=15)
    except Exception:
        pass

def tg_send_photo(token: str, chat_id: str, caption: str, png_path: str) -> None:
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(png_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": chat_id, "caption": caption}
            requests.post(url, data=data, files=files, timeout=30)
    except Exception:
        pass

# -----------------------------
# Tiny RSI strategy (built-in)
# -----------------------------
class IntradayRSIStrategy:
    def __init__(self, rsi_period=14, overbought=70, oversold=30,
                 sl_atr_mult=1.0, tgt_rr=1.5,
                 trail_start_atr=1.0, trail_atr_mult=1.0,
                 reentry_max=2, reentry_cooldown=3):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.sl_atr_mult = sl_atr_mult
        self.tgt_rr = tgt_rr
        self.trail_start_atr = trail_start_atr
        self.trail_atr_mult = trail_atr_mult
        self.reentry_max = reentry_max
        self.reentry_cooldown = reentry_cooldown

    @staticmethod
    def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        hl = (high - low).abs()
        hc = (high - close.shift()).abs()
        lc = (low - close.shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=n).mean()

    def _rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = (delta.clip(lower=0)).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return rsi.fillna(50)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["rsi"] = self._rsi(d["Close"])
        d["atr"] = self._atr(d, n=max(14, self.rsi_period))
        d["buy_sig"] = (d["rsi"] < self.oversold) & (d["rsi"].shift(1) >= self.oversold)
        d["sell_sig"] = (d["rsi"] > self.overbought) & (d["rsi"].shift(1) <= self.overbought)
        return d

# -----------------------------
# Backtest (index-level) with SL/Target/Trail/Re-entry
# -----------------------------
@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    entry: float
    exit_time: Optional[pd.Timestamp] = None
    exit: Optional[float] = None
    reason: str = ""
    pnl: float = 0.0

def run_bt(df: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame, pd.Series]:
    st = IntradayRSIStrategy(
        rsi_period=cfg.get("rsi_len", 14),
        overbought=cfg.get("rsi_sell", 70),
        oversold=cfg.get("rsi_buy", 30),
        sl_atr_mult=cfg.get("sl_atr_mult", 1.0),
        tgt_rr=cfg.get("tgt_rr", 1.5),
        trail_start_atr=cfg.get("trail_start_atr", 1.0),
        trail_atr_mult=cfg.get("trail_atr_mult", 1.0),
        reentry_max=cfg.get("reentry_max", 2),
        reentry_cooldown=cfg.get("reentry_cooldown", 3),
    )
    d = st.prepare(df)

    qty = int(cfg.get("order_qty", 1))
    capital = float(cfg.get("capital_rs", 100000))
    tz = cfg.get("tz", "Asia/Kolkata")

    position = "FLAT"
    entry_px = stop = target = np.nan
    trail_active = False
    re_count = 0
    last_exit_idx = -10**9
    trades: List[Trade] = []
    equity = capital
    eq_curve = []

    idx = list(d.index)
    for i, ts in enumerate(idx):
        row = d.loc[ts]
        px = float(row["Close"])
        atr = float(row.get("atr", 0) or 0)

        # trail update
        if position != "FLAT" and trail_active:
            if position == "LONG":
                stop = max(stop, px - st.trail_atr_mult * atr)
            else:
                stop = min(stop, px + st.trail_atr_mult * atr)

        # exits
        did_exit = False
        if position == "LONG":
            if row["Low"] <= stop:
                trades[-1].exit_time = ts
                trades[-1].exit = stop
                trades[-1].reason = "STOP"
                trades[-1].pnl = (stop - entry_px) * qty
                equity += trades[-1].pnl
                position = "FLAT"; did_exit = True
            elif row["High"] >= target:
                trades[-1].exit_time = ts
                trades[-1].exit = target
                trades[-1].reason = "TARGET"
                trades[-1].pnl = (target - entry_px) * qty
                equity += trades[-1].pnl
                position = "FLAT"; did_exit = True
            else:
                if (px - entry_px) >= st.trail_start_atr * atr:
                    trail_active = True

        elif position == "SHORT":
            if row["High"] >= stop:
                trades[-1].exit_time = ts
                trades[-1].exit = stop
                trades[-1].reason = "STOP"
                trades[-1].pnl = (entry_px - stop) * qty
                equity += trades[-1].pnl
                position = "FLAT"; did_exit = True
            elif row["Low"] <= target:
                trades[-1].exit_time = ts
                trades[-1].exit = target
                trades[-1].reason = "TARGET"
                trades[-1].pnl = (entry_px - target) * qty
                equity += trades[-1].pnl
                position = "FLAT"; did_exit = True
            else:
                if (entry_px - px) >= st.trail_start_atr * atr:
                    trail_active = True

        if did_exit:
            last_exit_idx = i

        # entries
        if position == "FLAT":
            ok_cd = (i - last_exit_idx) >= st.reentry_cooldown
            can_re = (re_count < st.reentry_max) or (st.reentry_max == 0)

            if ok_cd and can_re and bool(row["buy_sig"]):
                position = "LONG"; entry_px = px
                stop = entry_px - st.sl_atr_mult * atr
                target = entry_px + st.tgt_rr * (entry_px - stop)
                trail_active = False
                trades.append(Trade("LONG", ts, entry_px))
                if last_exit_idx > -10**8: re_count += 1

            elif ok_cd and can_re and bool(row["sell_sig"]):
                position = "SHORT"; entry_px = px
                stop = entry_px + st.sl_atr_mult * atr
                target = entry_px - st.tgt_rr * (stop - entry_px)
                trail_active = False
                trades.append(Trade("SHORT", ts, entry_px))
                if last_exit_idx > -10**8: re_count += 1

        # reset re-entries each new day
        if i > 0:
            prev = pd.Timestamp(idx[i-1]).tz_localize(None).date()
            cur = pd.Timestamp(ts).tz_localize(None).date()
            if prev != cur:
                re_count = 0

        eq_curve.append(equity)

    # results
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    eq_ser = pd.Series(eq_curve, index=d.index, name="equity")

    # metrics (minimal, independent)
    if trades_df.empty:
        metrics = {
            "n_trades": 0, "win_rate": 0.0, "roi_pct": 0.0, "profit_factor": 0.0,
            "rr": 0.0, "max_dd_pct": 0.0, "time_dd_bars": 0, "sharpe_ratio": 0.0
        }
    else:
        wins = (trades_df["pnl"] > 0).sum()
        losses = (trades_df["pnl"] < 0).sum()
        gross_p = trades_df["pnl"].clip(lower=0).sum()
        gross_l = -trades_df["pnl"].clip(upper=0).sum()
        profit_factor = (gross_p / gross_l) if gross_l > 0 else float("inf")
        roi_pct = (eq_ser.iloc[-1] - capital) / capital * 100.0
        win_rate = wins / len(trades_df) * 100.0

        # drawdown
        roll_max = eq_ser.cummax()
        dd = (eq_ser - roll_max) / roll_max
        max_dd_pct = dd.min() * 100.0 if len(dd) else 0.0
        time_dd_bars = (dd < 0).astype(int).groupby((dd >= 0).astype(int).diff().ne(0).cumsum()).transform("size")
        time_dd_bars = int(time_dd_bars.max()) if len(time_dd_bars) else 0

        # R:R
        avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean() if wins else 0.0
        avg_loss = -trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean() if losses else 0.0
        rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        # Sharpe (very rough, per-bar)
        rets = pd.Series(eq_ser).pct_change().dropna()
        sharpe = (rets.mean() / (rets.std() + 1e-9)) * np.sqrt(252 * 78) if len(rets) else 0.0

        metrics = {
            "n_trades": int(len(trades_df)),
            "win_rate": float(round(win_rate, 2)),
            "roi_pct": float(round(roi_pct, 2)),
            "profit_factor": float(round(profit_factor, 2)) if math.isfinite(profit_factor) else 0.0,
            "rr": float(round(rr, 2)),
            "max_dd_pct": float(round(max_dd_pct, 2)),
            "time_dd_bars": int(time_dd_bars),
            "sharpe_ratio": float(round(sharpe, 2)),
        }

    return metrics, trades_df, eq_ser

# -----------------------------
# Data fetch (prefer your data_io)
# -----------------------------
def get_candles(index_token: int, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """
    Delegates to data_io.get_index_candles if available.
    Otherwise returns an empty frame (runner will still create 'no signals' report).
    """
    try:
        from data_io import get_index_candles  # your repo helper
        return get_index_candles(index_token=index_token, timeframe=timeframe, start=start, end=end)
    except Exception as e:
        print("WARN: data_io.get_index_candles unavailable ->", e)
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"]).set_index(
            pd.to_datetime([])
        )

# -----------------------------
# Charts
# -----------------------------
def save_curve_charts(eq: pd.Series, out_dir: pathlib.Path):
    if eq is None or eq.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Equity
    plt.figure()
    eq.plot()
    plt.title("Equity Curve")
    plt.tight_layout()
    p1 = out_dir / "equity_curve.png"
    plt.savefig(p1); plt.close()

    # Drawdown
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    plt.figure()
    dd.plot()
    plt.title("Drawdown")
    plt.tight_layout()
    p2 = out_dir / "drawdown.png"
    plt.savefig(p2); plt.close()
    return str(p1), str(p2)

# -----------------------------
# Main
# -----------------------------
def main():
    rdir = pathlib.Path("reports"); rdir.mkdir(exist_ok=True, parents=True)

    cfg = load_yaml_cfg("config.yaml")
    intr = cfg.get("intraday_options", {}) or {}

    # Inputs (from workflow env if present)
    start = os.getenv("START_DATE", (dt.date.today() - dt.timedelta(days=30)).isoformat())
    end   = os.getenv("END_DATE", dt.date.today().isoformat())
    timeframe = os.getenv("TIMEFRAME", intr.get("timeframe", "5minute"))
    token = int(os.getenv("INDEX_TOKEN", intr.get("index_token", 260105)))  # BANKNIFTY default

    print(f"🔧 Using: token={token} {timeframe} {start}->{end}")

    # Download candles
    df = get_candles(index_token=token, timeframe=timeframe, start=start, end=end)
    if df is None or df.empty:
        note = "No data; runner fallback."
        # write minimal artifacts so Telegram never shows 'No summary'
        (rdir / "latest_signals.csv").write_text("timestamp,action,symbol,price,qty,pnl\n", encoding="utf-8")
        json.dump(
            {
                "n_trades": 0, "win_rate": 0.0, "roi_pct": 0.0, "profit_factor": 0.0,
                "rr": 0.0, "max_dd_pct": 0.0, "time_dd_bars": 0, "sharpe_ratio": 0.0, "note": note,
            },
            open(rdir / "metrics.json", "w"), indent=2
        )
        json.dump({"timestamp": dt.datetime.utcnow().isoformat() + "Z", "summary": note, "config": {}},
                  open(rdir / "latest.json", "w"), indent=2)
        print("No data frame available; wrote minimal reports.")
        # try telegram
        tg_send_text(os.getenv("TELEGRAM_BOT_TOKEN",""), os.getenv("TELEGRAM_CHAT_ID",""),
                     f"📊 Intraday Summary\n• Trades: 0\n• Note: {note}")
        return

    # Ensure index
    if "Date" in df.columns:
        df = df.rename(columns={"Date":"timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    # Run BT
    metrics, trades_df, eq = run_bt(df, {
        "tz": cfg.get("tz", "Asia/Kolkata"),
        "order_qty": intr.get("order_qty", cfg.get("order_qty", 1)),
        "capital_rs": cfg.get("capital_rs", 100000),

        "rsi_len": intr.get("rsi_len", 14),
        "rsi_buy": intr.get("rsi_buy", 30),
        "rsi_sell": intr.get("rsi_sell", 70),

        "sl_atr_mult": intr.get("sl_atr_mult", 1.0),
        "tgt_rr": intr.get("tgt_rr", 1.5),
        "trail_start_atr": intr.get("trail_start_atr", 1.0),
        "trail_atr_mult": intr.get("trail_atr_mult", 1.0),

        "reentry_max": intr.get("reentry_max", 2),
        "reentry_cooldown": intr.get("reentry_cooldown", 3),
    })

    # Save artifacts
    trades_csv = rdir / "latest_signals.csv"
    if trades_df.empty:
        trades_csv.write_text("timestamp,action,symbol,price,qty,pnl\n", encoding="utf-8")
    else:
        t = trades_df.copy()
        t["timestamp"] = t["entry_time"].astype(str)
        t["action"] = t["side"]
        t["symbol"] = "INDEX"
        t["price"] = t["entry"]
        t["qty"] = 1
        t["pnl"] = t["pnl"]
        t[["timestamp","action","symbol","price","qty","pnl"]].to_csv(trades_csv, index=False)

    json.dump(metrics, open(rdir / "metrics.json", "w"), indent=2)
    json.dump(
        {"timestamp": dt.datetime.utcnow().isoformat()+"Z", "summary": "Intraday run", "config": intr},
        open(rdir / "latest.json", "w"), indent=2
    )

    # Charts
    pics = save_curve_charts(eq, rdir)

    # Telegram Summary
    token = os.getenv("TELEGRAM_BOT_TOKEN","")
    chat = os.getenv("TELEGRAM_CHAT_ID","")
    summary = (
        "📊 <b>Intraday Summary</b>\n"
        f"• Trades: <b>{metrics['n_trades']}</b>\n"
        f"• Win-rate: <b>{metrics['win_rate']}%</b>\n"
        f"• ROI: <b>{metrics['roi_pct']}%</b>\n"
        f"• Profit Factor: <b>{metrics['profit_factor']}</b>\n"
        f"• R:R: <b>{metrics['rr']}</b>\n"
        f"• Max DD: <b>{metrics['max_dd_pct']}%</b>\n"
        f"• Time DD (bars): <b>{metrics['time_dd_bars']}</b>\n"
        f"• Sharpe: <b>{metrics['sharpe_ratio']}</b>"
    )
    tg_send_text(token, chat, summary)
    if pics:
        p1, p2 = pics
        tg_send_photo(token, chat, "Equity curve", p1)
        tg_send_photo(token, chat, "Drawdown", p2)

    print("✅ Intraday run complete.")

if __name__ == "__main__":
    main()
