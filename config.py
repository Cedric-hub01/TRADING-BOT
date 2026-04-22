# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  RisenFX demo account  (TradeLocker backend)
# ─────────────────────────────────────────────────────────────────────────────

# ── RisenFX credentials ───────────────────────────────────────────────────────
EMAIL    = "hatgkcedric2@gmail.com"
PASSWORD = "Cedric12$$"
SERVER   = "RISENFX"

# ── TradeLocker REST base ─────────────────────────────────────────────────────
BASE_URL = "https://demo.tradelocker.com/backend-api"

# ── Twelve Data (real-time candles, primary feed) ─────────────────────────────
# Yahoo Finance has a ~15-minute delay which is fatal for 1-min trading.  Twelve
# Data serves real-time EURUSD candles.  Get a free API key at
# https://twelvedata.com/ and paste it below.
#
# Free plan: 8 req/min, 800 req/day.  Scanning every 60 s = 1440 req/day, so the
# bot WILL hit the daily cap after ~13 h of uptime — at that point market_data
# transparently falls back to Yahoo.  Leave this empty to force Yahoo-only mode.
TWELVE_DATA_API_KEY = ""

# ── Hard-coded account / instrument IDs (RisenFX demo) ────────────────────────
# Discovery endpoints work, but hard-coding avoids a round-trip every start-up
# and guarantees we hit the correct account/route combo for EURUSD.E.
ACCOUNT_ID    = 2111100      # internal account id
ACC_NUM       = 2            # value for the `accNum` header
INSTRUMENT_ID = 18670        # EURUSD.E tradableInstrumentId
ROUTE_TRADE   = 1864547      # execution route
ROUTE_INFO    = 1864530      # info/quote route

# ── Instrument ────────────────────────────────────────────────────────────────
SYMBOL    = "EURUSD"
TIMEFRAME = "5"          # 5-minute bars  (valid: "1","5","15","30","60","240","1D")

# ── Strategy selector (apply_best_strategy.py rewrites these 3 fields) ────────
# STRATEGY_TYPE  : key in strategy._BUILDERS (e.g. "ema_cross", "supertrend"…)
# STRATEGY_PARAMS: dict of params for that strategy
# DIRECTION      : "long" | "short" | "both"
STRATEGY_TYPE   = "ema_cross"
STRATEGY_PARAMS = {"fast": 8, "med": 21, "slow": 50, "trend_filter": True}
DIRECTION       = "long"

# ── ATR ───────────────────────────────────────────────────────────────────────
ATR_PERIOD  = 14
ATR_SL_MULT = 1.5    # stop-loss   = entry − 1.5 × ATR
ATR_TP_MULT = 3.0    # take-profit = entry + 3.0 × ATR  (2:1 RR)

# ── Position sizing ───────────────────────────────────────────────────────────
LOT_SIZE = 0.05      # standard lots

# ── Risk limits ───────────────────────────────────────────────────────────────
MAX_LOSS_PER_TRADE  =   5.0   # USD — skip trade if SL distance > this
DAILY_STOP          = -20.0   # USD — halt trading for the day
DAILY_TARGET        =  50.0   # USD — halt trading for the day
MAX_CONSEC_LOSSES   =   3     # consecutive losses before pausing

# ── Loop ──────────────────────────────────────────────────────────────────────
SCAN_INTERVAL  = 60    # seconds between scans  (also the SL/TP monitor interval)
CANDLES_NEEDED = 120   # bars to fetch (need 50+ for EMA50; 120 = buffer)

# ── EURUSD pip / USD constants ────────────────────────────────────────────────
PIP_SIZE       = 0.0001
PIP_VALUE_STD  = 10.0   # USD per pip per 1.0 standard lot
