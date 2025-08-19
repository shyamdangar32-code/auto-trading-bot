import pandas as pd

def build_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = df.copy()
    # Entry if trend up + momentum OK
    d["long_entry"] = (d["ema_f"] > d["ema_s"]) & (d["rsi"] < cfg["rsi_sell"]) & (d["adx"] > 18)
    # Exit if trend down or overheated
    d["long_exit"]  = (d["ema_f"] < d["ema_s"]) | (d["rsi"] > cfg["rsi_sell"])

    d["label"] = "HOLD"
    d.loc[d["long_entry"], "label"] = "BUY"
    d.loc[d["long_exit"],  "label"] = "EXIT"
    return d
