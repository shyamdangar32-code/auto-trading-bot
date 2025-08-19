import json, pandas as pd, requests as rq, streamlit as st

# === Fill these with your GitHub username & repo ===
USER = "YOUR_USER"
REPO = "auto-trading-bot"

RAW  = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/reports/latest.json"
CSV  = f"https://raw.githubusercontent.com/{USER}/{REPO}/main/reports/latest_signals.csv"

st.set_page_config(page_title="Quant TV", layout="wide")
st.title("📺 Quant TV — Live Strategy Monitor")

c1,c2,c3 = st.columns(3)
try:
    latest = rq.get(RAW, timeout=10).json()
    c1.metric("Total PnL", f"₹{latest['metrics']['total_PnL']}")
    c2.metric("Win rate", f"{latest['metrics']['win_rate']}%")
    c3.metric("Trades", latest['metrics']['n_trades'])
    st.caption(f"Last signal: **{latest['last_label']}** at {latest['last_row'].get('Date','')}")
except Exception as e:
    st.warning(f"Could not load latest.json ({e})")

st.divider()
try:
    df = pd.read_csv(CSV)
    st.line_chart(df.set_index("Date")["Close"])
    st.dataframe(df.tail(50), use_container_width=True)
except Exception as e:
    st.warning(f"Could not load latest_signals.csv ({e})")
