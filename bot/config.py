# bot/config.py
import os
import yaml

def _as_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in ("1","true","yes","y","on")

def _mask(k, v):
    if v is None:
        return None
    HIDE = ("TOKEN","KEY","SECRET","ACCESS")
    return "***" if any(h in k.upper() for h in HIDE) else v

def get_cfg():
    # 1) load YAML (repo defaults)
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f) or {}

    # 2) overlay secrets from env (GitHub Actions)
    # Telegram
    cfg["telegram_bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN") or cfg.get("telegram_bot_token", "")
    cfg["telegram_chat_id"]   = os.getenv("TELEGRAM_CHAT_ID")   or cfg.get("telegram_chat_id", "")

    # Zerodha
    cfg["zerodha_api_key"]     = os.getenv("ZERODHA_API_KEY")     or cfg.get("zerodha_api_key", "")
    cfg["zerodha_api_secret"]  = os.getenv("ZERODHA_API_SECRET")  or cfg.get("zerodha_api_secret", "")
    cfg["zerodha_access_token"]= os.getenv("ZERODHA_ACCESS_TOKEN")or cfg.get("zerodha_access_token", "")

    # optional boolean override from env
    if "SEND_ONLY_SIGNALS" in os.environ:
        cfg["send_only_signals"] = _as_bool(os.getenv("SEND_ONLY_SIGNALS"))

    # keep a small, masked echo for debugging
    safe = {k: _mask(k, v) for k, v in cfg.items()}
    print("CFG:", safe)

    return cfg
