import numpy as np
import pandas as pd

def backtest(df: pd.DataFrame, cfg: dict) -> dict:
    """Very light long-only backtest (one position at a time)."""
    entries = df.index[df["long_entry"]].tolist()
    exits   = df.index[df["long_exit"]].tolist()
    entries.sort(); exits.sort()

    # Pair entry→exit in order
    PnLs = []
    ei = xi = 0
    pos = None
    while ei < len(entries):
        e = entries[ei]
        # find the first exit after entry (or last row)
        while xi < len(exits) and exits[xi] <= e:
            xi += 1
        x = exits[xi] if xi < len(exits) else df.index[-1]
        buy  = float(df.loc[e, "Close"])
        sell = float(df.loc[x, "Close"])
        PnLs.append((sell - buy) * cfg["order_qty"])
        ei += 1; xi += 1

    total   = float(np.sum(PnLs)) if PnLs else 0.0
    winrate = float(np.mean([p>0 for p in PnLs]))*100 if PnLs else 0.0
    return {"total_PnL": round(total,2), "win_rate": round(winrate,1), "n_trades": len(PnLs)}
