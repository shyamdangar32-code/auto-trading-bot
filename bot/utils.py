# bot/utils.py
import os
import json
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ---------- filesystem helpers ----------

def ensure_dir(path: str) -> None:
    """Create a folder if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def _json_default(o):
    """Make pandas / numpy / datetime objects JSON-safe."""
    if isinstance(o, (pd.Timestamp, datetime.datetime, datetime.date)):
        return o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj, path: str) -> None:
    """Write JSON with safe conversions for pandas/numpy/datetime."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str):
    """Read JSON; return None if missing/invalid."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# ---------- telegram helper ----------

def send_telegram(message: str, token: str | None = None, chat_id: str | None = None) -> None:
    """
    Send a Telegram message. If token/chat_id are not provided, read
    them from environment variables:
      TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    """
    token = (token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()

    print("SEND ->", message)
    if not token or not chat_id:
        print("⚠️ Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
        return

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print("Telegram send failed:", repr(e))
