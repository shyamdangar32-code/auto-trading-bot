import os
import requests
import pandas as pd
from datetime import datetime

# === CONFIG ===
CFG = {
    "ema_period": 20,
    "rsi_period": 14,
    "symbol": "NIFTY",
}
OUT = "out"

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "7231058583"   # your chat ID


# === Utility functions ===
def add_indicators(df, cfg):
    # Example: add EMA + RSI
    df["EMA20"] = df["Close"].ewm(span=cfg["ema_period"]).mean()
    df["RSI"] = compute_rsi(df["Close"], cfg["rsi_period"])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def build_signals(df, cfg):
    # Simple EMA + RSI strategy
    df["label"] = "HOLD"
    df.loc[(df["Close"] > df["EMA20"]) & (df["RSI"] < 30), "label"] = "BUY"
    df.loc[(df["Close"] < df["EMA20"]) & (df["RSI"] > 70), "label"] = "SELL"
    return df

def backtest(df, cfg):
    # Just return dummy metrics for now
    return {"trades": len(df), "symbol": cfg["symbol"]}

def save_json(obj, path):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    import json
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        r = requests.post(url, json=payload)
        if r.status_code != 200:
            print("⚠️ Telegram error:", r.text)
    except Exception as e:
        print("⚠️ Telegram exception:", e)

def status_line(last, label):
    return f"{label} | {last['Date']} | Close={last['Close']}"


# === MAIN ===
def main():
    # 1) Load data
    df = pd.read_csv("data.csv")   # <-- Replace with your CSV or API data
    df["Date"] = pd.to_datetime(df["Date"])

    # 2) Features & signals
    df = add_indicators(df, CFG)
    df = build_signals(df, CFG)
    metrics = backtest(df, CFG)

    # 3) Outputs
    last = df.iloc[-1].copy()
    if "Date" in last.index:
        last["Date"] = str(last["Date"])

    latest_json_path = f"{OUT}/latest.json"
    prev = load_json(latest_json_path) or {}
    prev_label = prev.get("last_label")

    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "config": CFG,
        "metrics": metrics,
        "last_label": str(last.get("label", "")),
        "last_row": last.to_dict(),
    }

    save_json(payload, latest_json_path)
    df.tail(250).to_csv(f"{OUT}/latest_signals.csv", index=False)

    # 4) Logs + Alerts
    label = str(last.get("label", ""))
    print("📊 Backtest:", metrics)
    print(status_line(last, label))

    if label != prev_label and label != "HOLD":
        msg = status_line(last, label) + f"\nPnL: {metrics}"
        send_telegram(msg)
    else:
        print("ℹ️ No alert sent (unchanged or HOLD).")


if __name__ == "__main__":
    main()
