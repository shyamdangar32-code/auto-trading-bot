+++ live_runner.py
@@
+import os, json, time, yaml
+from datetime import datetime, timezone
+import pandas as pd
+
+# project modules
+from bot.data_io import prices
+from bot.indicators import add_indicators
+from bot.strategy import build_signals
+from bot.utils import send_telegram
+
+# ---- load config ----
+CFG = yaml.safe_load(open("config.yaml"))
+OUT_DIR = CFG.get("out_dir", "reports")
+os.makedirs(OUT_DIR, exist_ok=True)
+STATE_PATH = os.path.join(OUT_DIR, "live_state.json")
+
+
+def _load_state():
+    try:
+        return json.load(open(STATE_PATH))
+    except Exception:
+        return {"last_label": None}
+
+
+def _save_state(state: dict):
+    json.dump(state, open(STATE_PATH, "w"))
+
+
+def _serialize_row(row: pd.Series) -> dict:
+    d = dict(row)
+    if isinstance(d.get("Date"), (pd.Timestamp, )):
+        d["Date"] = d["Date"].isoformat()
+    return d
+
+
+def _status_line(df: pd.DataFrame, label: str) -> str:
+    last = df.iloc[-1]
+    return f"📈 {CFG['symbol']} | {str(last['Date'])[:19]} | Price: {round(float(last['Close']),2)} | Signal: {label}"
+
+
+def _backtest_simple(df: pd.DataFrame, qty: int = 1):
+    """Very small PnL summary from long_entry/long_exit columns."""
+    entries = df.index[df["long_entry"]].tolist()
+    exits   = df.index[df["long_exit"]].tolist()
+
+    # pair entries to exits in sequence
+    trades = []
+    e_i = x_i = 0
+    while e_i < len(entries) and x_i < len(exits):
+        if exits[x_i] <= entries[e_i]:
+            x_i += 1
+            continue
+        ei, xi = entries[e_i], exits[x_i]
+        buy  = float(df.loc[ei, "Close"])
+        sell = float(df.loc[xi, "Close"])
+        trades.append({"entry_i": int(ei), "exit_i": int(xi), "pnl": (sell - buy) * qty})
+        e_i += 1; x_i += 1
+
+    pnls = [t["pnl"] for t in trades if "pnl" in t]
+    total = float(sum(pnls)) if pnls else 0.0
+    win_rate = float(sum(p > 0 for p in pnls)) / len(pnls) * 100.0 if pnls else 0.0
+    return {"total_PnL": round(total, 2), "win_rate": round(win_rate, 1), "n_trades": len(pnls)}, trades
+
+
+def tick_once():
+    # 1) data
+    df = prices(
+        symbol=CFG["symbol"],
+        period=CFG["lookback"],
+        interval=CFG["interval"],
+        zerodha_enabled=CFG.get("zerodha_enabled", False),
+        zerodha_instrument_token=CFG.get("zerodha_instrument_token"),
+    )
+
+    # 2) indicators + signals
+    df = add_indicators(df, CFG)
+    df = build_signals(df, CFG)
+
+    # 3) quick backtest metrics (for dashboard)
+    metrics, trades = _backtest_simple(df, qty=CFG.get("order_qty", 1))
+
+    # 4) latest label
+    last = df.iloc[-1]
+    label = "BUY" if last["long_entry"] else ("EXIT" if last["long_exit"] else "HOLD")
+
+    # 5) write artifacts the Streamlit app reads
+    bundle = {
+        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
+        "config": {
+            **{k: v for k, v in CFG.items() if k not in {
+                "telegram_bot_token", "telegram_chat_id",
+                "zerodha_api_key", "zerodha_api_secret", "zerodha_access_token"
+            }},
+            "telegram_bot_token": "***",
+            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
+            "zerodha_api_key": "***",
+            "zerodha_api_secret": "***",
+            "zerodha_access_token": "***",
+        },
+        "metrics": metrics,
+        "last_signal": label,
+        "last_row": _serialize_row(last),
+    }
+    json.dump(bundle, open(os.path.join(OUT_DIR, "latest.json"), "w"), indent=2)
+    df.tail(250).to_csv(os.path.join(OUT_DIR, "latest_signals.csv"), index=False)
+
+    # 6) alert only on change and not HOLD
+    state = _load_state()
+    if label != "HOLD" and label != state.get("last_label"):
+        try:
+            send_telegram(_status_line(df, label))
+        except Exception as e:
+            print("Telegram error:", repr(e))
+    state["last_label"] = label
+    _save_state(state)
+
+    print("✅ tick:", _status_line(df, label), "|", metrics)
+    return label
+
+
+if __name__ == "__main__":
+    sleep_sec = int(os.getenv("LIVE_SLEEP_SEC", "300"))  # default 5 min
+    print("🔁 Live loop started… (CTRL+C to stop locally)")
+    while True:
+        try:
+            tick_once()
+        except Exception as e:
+            print("❌ cycle error:", repr(e))
+        time.sleep(sleep_sec)
