# runner.py
import os
import json
from bot.config import load_config
from bot.data_io import prices
from bot.backtest import run_backtest, save_reports

def main():
    CFG = load_config("config.yaml")
    out_dir = CFG.get("out_dir", "reports")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Data ----
    df = prices(
        symbol=CFG["symbol"],
        period=CFG.get("lookback", "1y"),
        interval=CFG.get("interval", "1d"),
        tz=CFG.get("tz", "Asia/Kolkata"),
        zerodha_enabled=CFG.get("zerodha_enabled", True),
        zerodha_instrument_token=CFG.get("zerodha_instrument_token")
    )

    print(f"🟢 Data OK: {len(df)} rows")

    # ---- Backtest with Re-entry + Trailing ----
    summary, trades_df, equity_ser = run_backtest(df, CFG)
    save_reports(out_dir, summary, trades_df, equity_ser)

    # ---- Console summary ----
    print("\n📊 Evaluation")
    print(f"• Trades      : {summary['n_trades']}")
    print(f"• Win-rate    : {summary['win_rate']}%")
    print(f"• ROI         : {summary['roi_pct']}%")
    print(f"• Max DD      : {summary['max_dd_pct']}%")
    print(f"• Time DD(bars): {summary['time_dd_bars']}")
    print(f"• R:R         : {summary['rr']}")
    if trades_df is not None and not trades_df.empty:
        last = trades_df.iloc[-1]
        print(f"\n🔔 Last trade: {last['side']} | entry {last['entry']:.2f} @ {last['entry_time']} "
              f"-> exit {last['exit']:.2f} ({last['reason']}) | PnL {last['pnl']:.2f}")

    # keep a quick snapshot for other tools
    with open(os.path.join(out_dir, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
