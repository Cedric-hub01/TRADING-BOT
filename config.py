# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  —  edit the three lines marked UPDATE to switch brokers
# ─────────────────────────────────────────────────────────────────────────────

# ── Broker credentials ────────────────────────────────────────────────────────
EMAIL    = "your@email.com"          # UPDATE — TradeLocker demo account email
PASSWORD = "your_password"           # UPDATE — TradeLocker demo account password
SERVER   = "OSP-DEMO"                # UPDATE — server tag shown on broker login page

# ── TradeLocker REST base ─────────────────────────────────────────────────────
BASE_URL = "https://demo.tradelocker.com/backend-api"

# ── Instrument ────────────────────────────────────────────────────────────────
SYMBOL    = "EURUSD"
TIMEFRAME = "5"          # 5-minute bars  (valid: "1","5","15","30","60","240","1D")

# ── EMA crossover ─────────────────────────────────────────────────────────────
EMA_FAST = 8
EMA_MED  = 21
EMA_SLOW = 50

# ── ATR ───────────────────────────────────────────────────────────────────────
ATR_PERIOD  = 14
ATR_SL_MULT = 1.5    # stop-loss  = entry − 1.5 × ATR
ATR_TP_MULT = 3.0    # take-profit = entry + 3.0 × ATR  (3:1 RR)

# ── Position sizing ───────────────────────────────────────────────────────────
LOT_SIZE = 0.05      # standard lots

# ── Risk limits ───────────────────────────────────────────────────────────────
MAX_LOSS_PER_TRADE  =   5.0   # USD — skip trade if SL distance > this
DAILY_STOP          = -20.0   # USD — halt trading for the day
DAILY_TARGET        =  50.0   # USD — halt trading for the day
MAX_CONSEC_LOSSES   =   3     # consecutive losses before pausing

# ── Loop ──────────────────────────────────────────────────────────────────────
SCAN_INTERVAL  = 60    # seconds between scans
CANDLES_NEEDED = 120   # how many bars to fetch (need 50+ for EMA50; 120 = buffer)

# ── EURUSD pip/USD constants ──────────────────────────────────────────────────
PIP_SIZE       = 0.0001
PIP_VALUE_STD  = 10.0   # USD per pip per 1.0 standard lot
