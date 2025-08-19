import os, json, pathlib, requests

def ensure_dir(p: str) -> None:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def send_telegram(text: str) -> None:
    tok = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    cid = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not tok or not cid:
        print("⚠️ Telegram not configured; skipping.")
        return
    try:
        url = f"https://api.telegram.org/bot{tok}/sendMessage"
        r = requests.post(url, json={"chat_id": cid, "text": text}, timeout=20)
        r.raise_for_status()
        print("📨 Telegram sent.")
    except Exception as e:
        print("Telegram error:", repr(e))
