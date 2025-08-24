import os, sys
from kiteconnect import KiteConnect

def snip(s):  # safe, shows only length & ends
    return f"len={len(s)}, head={s[:3]!r}, tail={s[-3:]!r}"

api_key = os.getenv("ZERODHA_API_KEY", "").strip()
access  = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()

if not api_key or not access:
    print("❌ Missing envs.")
    print("ZERODHA_API_KEY:", snip(api_key))
    print("ZERODHA_ACCESS_TOKEN:", snip(access))
    sys.exit(1)

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access)

try:
    profile = kite.profile()
    print("✅ Kite profile OK:", profile.get("user_id"))
except Exception as e:
    print("❌ Kite auth failed:", e)
    print("ZERODHA_API_KEY:", snip(api_key))
    print("ZERODHA_ACCESS_TOKEN:", snip(access))
    sys.exit(1)
