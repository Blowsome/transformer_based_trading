"""All settings for the pipeline in one place."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
LOG_DIR = OUT_DIR / "logs"

# ---- Dates (splits are by the days whose returns are predicted) ----
START = "2015-01-01"
TRAIN_END = "2024-12-31"
VAL_START, VAL_END = "2025-01-01", "2025-12-31"   # used for early stopping and model choices
TEST_START = "2026-01-01"
TEST_END = date.today().isoformat()

# ---- Universe: US common stocks on Nasdaq, NYSE, NYSE American, NYSE Arca, Cboe BZX, IEX ----
TICKER_SOURCES = {
    "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}
EXCLUDE_NAME_PATTERN = (r"warrant|\bunits?\b|\bright|preferred|depositary|\bnotes?\b|debenture|% |\bfund\b|"
                        r"\betn\b|exchange traded")
EXCLUDE_SYMBOL_PATTERN = r"\$|\.(W|WS|U|R|RT|WI)$"   # preferreds ($), warrants, units, rights
# A stock may list at any time; it is used from the day it has WARMUP + WINDOW days of history.
# It must still be listed today (Yahoo has no data for delisted stocks: survivorship bias).
MAX_MISSING_FRAC = 0.02         # max fraction of missing trading days after its first day
SPIKE_LOG_RET = 0.7             # bad data: a ~2x jump that reverses the next day ...
MAX_LOG_RET = 2.3               # ... or any one-day move above ~10x

# ---- Tradable on a given day (used for training rows, labels, and the backtest) ----
MIN_PRICE = 5.0                 # close >= $5 on the decision day
MIN_DOLLAR_VOL = 1e6            # 20-day median dollar volume >= $1M
                                # ... and at least WARMUP + WINDOW days of history, so every input is defined

# ---- Download ----
CHUNK_SIZE = 100
SLEEP_BETWEEN_CHUNKS = 1.0

# ---- Target ----
HORIZON = 5                     # predict the next 5-day return, as a rank among tradable stocks that day

# ---- Features / windows ----
WINDOW = 20                     # days of features the model sees
WARMUP = 60                     # days needed by the longest rolling feature
STOCK_FEATURES = [              # ranked across stocks each day, scaled to -0.5..0.5
    "ret1", "ret5", "ret10", "ret20", "ret60",
    "vol5", "vol20", "vol60",
    "hl_range", "co_ret", "gap",
    "vol_ratio5", "vol_ratio20",
    "ma5_gap", "ma20_gap", "ma60_gap",
    "high20", "low20", "high60", "low60",
    "max_ret20", "min_ret20", "dollar_vol",
]
MARKET_FEATURES = [             # same for every stock that day; z-scored with training-period stats
    "mkt_ret1", "mkt_ret5", "mkt_ret20", "mkt_vol20", "mkt_dispersion",
]
# Off: with ~600 training days, market features let models memorize individual days
# (LightGBM validation IC 0.024 with them vs 0.048 without)
USE_MARKET_FEATURES = False
FEATURE_CLIP = 5.0

# ---- Model ----
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
DROPOUT = 0.2
CROSS_STOCK = True              # extra attention layer across all stocks of the same day (MASTER-style)

# ---- Training (one batch = all tradable stocks of one day) ----
SEED = 42
N_SEEDS = 3                     # models in the ensemble (about 30-90 min each on an M1)
LOSS = "ic"                     # "ic" (negative daily correlation), "listnet", or "mse"
MSE_WEIGHT = 0.1                # small MSE term added to the "ic" loss to keep predictions on scale
LISTNET_TEMP = 0.05
TRAIN_STOCKS_PER_DAY = 1500     # each training step uses a random subset of the day's stocks (None = all);
                                # a different subset every epoch makes memorizing harder and steps faster
LR = 3e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 20
PATIENCE = 4                    # epochs without a better validation IC before stopping
GRAD_CLIP = 1.0
LOG_EVERY = 50                  # steps between progress updates

# ---- Regularization comparison (one seed each, judged on validation IC; the winner trains the ensemble) ----
REGULARIZATION_OPTIONS = {
    "reg_current": {"dropout": DROPOUT, "weight_decay": WEIGHT_DECAY},
    "reg_strong": {"dropout": 0.3, "weight_decay": 1e-2},
}

# ---- Baselines ----
ARIMA_ORDER = (1, 0, 1)
ARIMA_WORKERS = 4
LGBM_PARAMS = {
    "objective": "regression", "learning_rate": 0.02, "num_leaves": 63, "min_data_in_leaf": 1000,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
    "verbose": -1, "num_threads": 4,
}
LGBM_ROUNDS = 2000
LGBM_EARLY_STOP = 100
