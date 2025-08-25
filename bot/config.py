# bot/config.py
"""
Config loader for the trading bot.

- Reads config.yaml
- Overrides sensitive values with environment variables
- ALWAYS returns a plain dict (NOT SimpleNamespace), so callers can use .get()
"""

from __future__ import annotations

import os
import yaml
from typing import Any, Dict


# -------------------------
# internal helpers
# -------------------------

def _read_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("config.yaml must contain a YAML mapping (dict at top level).")
        return data
    except FileNotFoundError:
        # empty baseline if file does not exist
        return {}
    except Exception as e:
        raise RuntimeError(f"Failed to read YAML config at {path}: {e}")


def _set_if_env(cfg: Dict[str, Any], key: str, env_name: str) -> None:
    """
    Write cfg[key] = os.getenv(env_name) if present (non-empty),
    otherwise keep existing cfg[key] untouched.
    """
    val = os.getenv(env_name)
    if val is not None and val != "":
        cfg[key] = val


# -------------------------
# public API
# -------------------------

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration as a dict with env overrides.

    Keys exposed at top-level (for convenience with .get()):
      - ZERODHA_API_KEY
      - ZERODHA_API_SECRET
      - ZERODHA_ACCESS_TOKEN
      - TELEGRAM_BOT_TOKEN
      - TELEGRAM_CHAT_ID

    Also writes Telegram overrides into intraday_options.* if that section exists.
    """
    cfg = _read_yaml(path)

    # Ensure sections exist
    intraday = cfg.get("intraday_options") or {}
    cfg["intraday_options"] = intraday

    # --- Zerodha secrets (top-level convenience keys) ---
    # (these are expected to come from GitHub Secrets in Actions)
    if "ZERODHA_API_KEY" not in cfg:
        cfg["ZERODHA_API_KEY"] = None
    if "ZERODHA_API_SECRET" not in cfg:
        cfg["ZERODHA_API_SECRET"] = None
    if "ZERODHA_ACCESS_TOKEN" not in cfg:
        cfg["ZERODHA_ACCESS_TOKEN"] = None

    _set_if_env(cfg, "ZERODHA_API_KEY", "ZERODHA_API_KEY")
    _set_if_env(cfg, "ZERODHA_API_SECRET", "ZERODHA_API_SECRET")
    _set_if_env(cfg, "ZERODHA_ACCESS_TOKEN", "ZERODHA_ACCESS_TOKEN")

    # --- Telegram (top-level convenience + nested intraday_options) ---
    if "TELEGRAM_BOT_TOKEN" not in cfg:
        cfg["TELEGRAM_BOT_TOKEN"] = intraday.get("telegram_bot_token")
    if "TELEGRAM_CHAT_ID" not in cfg:
        cfg["TELEGRAM_CHAT_ID"] = intraday.get("telegram_chat_id")

    _set_if_env(cfg, "TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN")
    _set_if_env(cfg, "TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID")

    # Mirror into intraday_options if present
    if cfg.get("TELEGRAM_BOT_TOKEN"):
        intraday["telegram_bot_token"] = cfg["TELEGRAM_BOT_TOKEN"]
    if cfg.get("TELEGRAM_CHAT_ID"):
        intraday["telegram_chat_id"] = cfg["TELEGRAM_CHAT_ID"]

    # Return a plain dict (callers will use cfg.get("KEY"))
    return cfg


def debug_fingerprint(cfg: Dict[str, Any]) -> str:
    """
    Safe one-line fingerprint of sensitive fields (lengths only).
    Useful for logging in CI without exposing secrets.
    """
    def L(v: Any) -> int:
        return len(v) if isinstance(v, str) else 0

    return (
        f"API_KEY:{L(cfg.get('ZERODHA_API_KEY'))} "
        f"API_SECRET:{L(cfg.get('ZERODHA_API_SECRET'))} "
        f"ACCESS_TOKEN:{L(cfg.get('ZERODHA_ACCESS_TOKEN'))} "
        f"TG_TOKEN:{L(cfg.get('TELEGRAM_BOT_TOKEN'))} "
        f"TG_CHAT:{L(str(cfg.get('TELEGRAM_CHAT_ID') or ''))}"
    )
