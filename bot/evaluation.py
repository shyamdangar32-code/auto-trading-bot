# bot/evaluation.py
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_equity_and_drawdown(equity: pd.Series, out_dir: str) -> dict:
    """
    Saves equity and drawdown charts. Returns dict with file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Equity
    eq_png = os.path.join(out_dir, "equity_curve.png")
    plt.figure()
    equity.plot()
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity (₹)")
    plt.tight_layout()
    plt.savefig(eq_png)
    plt.close()

    # Drawdown
    peak = equity.cummax()
    dd = equity - peak
    dd_png = os.path.join(out_dir, "drawdown.png")
    plt.figure()
    dd.plot()
    plt.title("Drawdown (₹)")
    plt.xlabel("Trade #")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(dd_png)
    plt.close()

    return {"equity_curve_png": eq_png, "drawdown_png": dd_png}

def write_quick_report(summary: dict, trades: pd.DataFrame, out_dir: str) -> str:
    """
    Writes a markdown summary with key metrics and the first few trades.
    """
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "report.md")

    lines = [
        "# Backtest Report",
        "",
        "## Summary",
        f"- Trades: **{summary.get('n_trades', 0)}**",
        f"- Win rate: **{summary.get('win_rate', 0)}%**",
        f"- ROI: **{summary.get('roi_pct', 0)}%**",
        f"- Profit Factor: **{summary.get('profit_factor', '—')}**",
        f"- R:R: **{summary.get('rr', '—')}**",
        f"- Max DD: **{summary.get('max_dd_pct', 0)}%**",
        f"- Time in DD (bars): **{summary.get('time_dd_bars', 0)}**",
        f"- Sharpe: **{summary.get('sharpe_ratio', '—')}**",
        "",
        "## Sample trades",
    ]
    if trades is not None and not trades.empty:
        head = trades.head(10).to_markdown(index=False)
        lines.append(head)
    else:
        lines.append("_No trades._")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path
