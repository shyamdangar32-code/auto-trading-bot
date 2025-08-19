import itertools, json, yaml
from bot.data_io import yahoo_prices
from bot.indicators import add_indicators
from bot.strategy import build_signals
from bot.backtest import backtest
from bot.utils import ensure_dir

cfg = yaml.safe_load(open("config.yaml", "r"))

grid = {
    "ema_fast": [13, 21, 34],
    "ema_slow": [50, 89],
    "rsi_buy":  [25, 30],
    "rsi_sell": [65, 70],
}

def score(c):
    cfull = {**cfg, **c}
    df = yahoo_prices(cfull["symbol"], cfull["lookback"], cfull["interval"])
    df = add_indicators(df, cfull)
    df = build_signals(df, cfull)
    m  = backtest(df, cfull)
    return m["total_PnL"], m

best = None
best_cfg = None
for values in itertools.product(*grid.values()):
    c = dict(zip(grid.keys(), values))
    s, m = score(c)
    if (best is None) or (s > best[0]):
        best = (s, m)
        best_cfg = c

print("Best:", best_cfg, best)

# write artifact
ensure_dir("reports")
with open("reports/opt_latest.json","w") as f:
    json.dump({"best_cfg":best_cfg, "metrics":best[1]}, f, indent=2)

# update config.yaml in-place (PR will capture this change)
cfg.update(best_cfg)
yaml.safe_dump(cfg, open("config.yaml","w"))
