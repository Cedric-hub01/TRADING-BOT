# EURUSD Trading Bot (RisenFX / TradeLocker)

Live bot + comprehensive backtester for EURUSD on a RisenFX demo account.

## Architecture

```
bot.py                  entry point — 60 s loop, monitor-or-scan
 ├ tradelocker.py       REST client (login, positions, market orders)
 ├ strategy.py          dispatches signals by STRATEGY_TYPE / PARAMS / DIRECTION
 │   └ backtest_strategies.py   shared signal library (used by bot AND backtest)
 ├ market_data.py       Yahoo Finance candles + current price (RisenFX history API broken)
 ├ risk.py              daily PnL / consec-loss / per-trade risk gate
 └ config.py            credentials + account/route IDs + strategy selection

backtest.py             sweeps every strategy × params × direction, writes leaderboards
apply_best_strategy.py  reads best_strategy.json and rewrites the 3 strategy keys
```

### SL/TP approach (why no pending orders)

RisenFX demo does not accept SL/TP attached to market orders (PATCH on positions
returns 404).  Pending stop/limit orders are only queued — they can miss fills.
The bot therefore **monitors price every 60 s and closes via an opposite-side
market order when price crosses the SL/TP level**, guaranteeing real execution.

## Running the bot

```bash
pip install -r requirements.txt
python bot.py
```

`config.py` is already populated with the RisenFX demo credentials, account
`2111100` / `accNum 2`, `EURUSD.E` instrument id `18670`, and route `1864547`.

## Backtest workflow

Because Yahoo is blocked in some sandboxes, run this **locally** where `yfinance`
can reach Yahoo.

```bash
# 1. Sweep every strategy × direction and rank them
python backtest.py --interval 5m --period 60d     # ~60 days of 5-min bars  (recommended)
# alternatives:
#   --interval 1m  --period 7d         (Yahoo's max resolution window)
#   --interval 15m --period 60d        (smoother, fewer trades)
#   --interval 1h  --period 730d       (2 years of hourly)

# 2. Inspect the results
#    - stdout             : six leaderboards (composite, net$, PF, win%, sharpe, DD)
#    - backtest_results.csv : every run (417 rows for the default grid)
#    - backtest_top10.json  : machine-readable top 10
#    - best_strategy.json   : the single winner by composite score

# 3. Install the winner into the live bot
python apply_best_strategy.py
#    Rewrites STRATEGY_TYPE / STRATEGY_PARAMS / DIRECTION in config.py and
#    writes a config.py.bak in case you want to revert.

# 4. Restart the bot — it will trade the new strategy
python bot.py
```

### Strategies evaluated

| Family | Combos |
| ------ | ------ |
| EMA crossovers (± EMA-slow trend filter) | ~80 |
| MACD (4 parameter sets) | 4 |
| Supertrend (period × mult) | 9 |
| Ichimoku TK-cross | 3 |
| RSI mean-reversion | 9 |
| Bollinger reversion & breakout | 12 |
| Stochastic | 4 |
| EMA + ADX filter | 12 |
| London / NY session breakouts | 6 |

Each × `{long, short, both}` → **~420 runs**.  Exits use the same ATR-based
SL/TP (1.5× / 3.0×) the live bot uses, plus realistic slippage (0.2 pip) and
round-trip spread (0.5 pip), so ranking reflects an *actually executable* edge.

### Yahoo Finance data caveat

1-minute bars are capped at ~7 days of history by Yahoo.  For a multi-month
window the best resolutions are 5-minute (60 days) or 15-minute (60 days).
Hourly goes back 2 years.  The `--interval` / `--period` CLI flags let you
pick the trade-off.
