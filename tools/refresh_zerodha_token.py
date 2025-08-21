import os
from kiteconnect import KiteConnect

def refresh_zerodha_token():
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    request_token = os.getenv("ZERODHA_REQUEST_TOKEN")

    kite = KiteConnect(api_key=api_key)

    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]

        print("✅ New Zerodha access_token generated:", access_token)

        # Save token to file for GitHub Actions use
        with open("zerodha_access_token.txt", "w") as f:
            f.write(access_token)

    except Exception as e:
        print("❌ Error refreshing token:", str(e))

if __name__ == "__main__":
    refresh_zerodha_token()
