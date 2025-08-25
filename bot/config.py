# bot/config.py
#
# Central configuration loader for all runners.
# - Reads config.yaml from repo root
# - Overlays with environment variables (GitHub Secrets)
# - Returns a SimpleNamespace for dot-access (cfg.intraday_options.lots, etc.)

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


# ---------- helpers ----------

def _to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """
    Convert nested dicts to SimpleNamespace so we can do cfg.foo.bar.
    Lists/datatypes other than dict are left as-is.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d  # type: ignore[return-value]


def _merge_env_overrides(cfg: Dict[str, Any]) -> None:
    """
    Overlay pieces that should come from environment variables / secrets.
    We do this in-place on cfg.
    """
    # Zerodha secrets
    z_api_key = (os.getenv("ZERODHA_API_KEY") or "").strip()
    z_api_secret = (os.getenv("ZERODHA_API_SECRET") or "").strip()
    z_access = (os.getenv("ZERODHA_ACCESS_TOKEN") or "").strip()

    # Telegram secrets
    tg_token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    tg_chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

    # Where to store secrets in cfg:
    # We keep a top-level "secrets" bag and also mirror into strategy blocks that need them.
    secrets = cfg.setdefault("secrets", {})
    if z_api_key:
        secrets["ZERODHA_API_KEY"] = z_api_key
    if z_api_secret:
        secrets["ZERODHA_API_SECRET"] = z_api_secret
    if z_access:
        secrets["ZERODHA_ACCESS_TOKEN"] = z_access

    if tg_token:
        secrets["TELEGRAM_BOT_TOKEN"] = tg_token
    if tg_chat_id:
        secrets["TELEGRAM_CHAT_ID"] = tg_chat_id

    # Also mirror Telegram into intraday_options section (if present)
    io = cfg.get("intraday_options") or {}
    if tg_token:
        io["telegram_bot_token"] = tg_token
    if tg_chat_id:
        io["telegram_chat_id"] = tg_chat_id
    cfg["intraday_options"] = io


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config.yaml missing at: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("config.yaml root must be a mapping/object")
        return data


# ---------- public API ----------

def load_config(config_path: str | Path = None) -> SimpleNamespace:
    """
    Load config.yaml + env overlays and return a SimpleNamespace.
    Usage:
        from bot.config import load_config
        cfg = load_config()
        print(cfg.tz)
        print(cfg.intraday_options.underlying)
        print(cfg.secrets.ZERODHA_API_KEY)  # if provided via secrets/env
    """
    # Resolve repo root relative to this file: repo_root / "config.yaml"
    repo_root = Path(__file__).resolve().parents[1]  # goes: bot/ -> repo root
    path = Path(config_path) if config_path else (repo_root / "config.yaml")

    cfg = _load_yaml(path)

    # Ensure defaults exist for expected sections
    cfg.setdefault("intraday_options", {})
    cfg.setdefault("secrets", {})

    # Overlay with environment variables (GitHub Secrets)
    _merge_env_overrides(cfg)

    # Return as namespace for dot-access
    return _to_namespace(cfg)


# Optional tiny utility: make sure output dirs exist
def ensure_dir(p: str | Path) -> Path:
    pth = Path(p)
    pth.mkdir(parents=True, exist_ok=True)
    return pth


if __name__ == "__main__":
    # quick self-test
    try:
        c = load_config()
        print("✅ Loaded config.")
        # print some key bits safely
        print("tz:", getattr(c, "tz", None))
        if hasattr(c, "intraday_options"):
            print("underlying:", getattr(c.intraday_options, "underlying", None))
            print("interval:", getattr(c.intraday_options, "interval", None))
        if hasattr(c, "secrets"):
            ak = getattr(c.secrets, "ZERODHA_API_KEY", "")
            at = getattr(c.secrets, "ZERODHA_ACCESS_TOKEN", "")
            print("api_key_len:", len(ak), "access_token_len:", len(at))
    except Exception as e:
        print("❌ Config load failed:", e)
        sys.exit(1)
