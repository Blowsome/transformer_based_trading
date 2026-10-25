"""All settings for the pipeline in one place."""
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"

# ---- Dates (splits are by the date of the day being predicted) ----
START = "2023-01-01"
TRAIN_END = "2025-09-30"
VAL_START, VAL_END = "2025-10-01", "2025-12-31"   # used only for early stopping
TEST_START = "2026-01-01"
TEST_END = date.today().isoformat()

# ---- Universe ----
TICKER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
EXCLUDE_NAME_PATTERN = r"warrant|\bunits?\b|\bright|preferred|depositary|\bnotes?\b|debenture|% "
MAX_FIRST_DATE = "2023-01-10"   # must already trade in the first week of 2023
MAX_MISSING_FRAC = 0.02         # max fraction of missing trading days
MIN_MEAN_CLOSE = 1.0            # drop penny stocks (training period)
MIN_MEDIAN_DOLLAR_VOL = 2.5e5   # drop illiquid stocks (training period)

# ---- Download ----
CHUNK_SIZE = 100
SLEEP_BETWEEN_CHUNKS = 1.0

# ---- Features / windows ----
WINDOW = 60
WARMUP = 20                     # days needed by 20-day rolling features
FEATURES = [
    "ret", "hl_range", "co_ret", "vol_ratio",
    "ma20_gap", "ret5", "vol20", "gap",
]
FEATURE_CLIP = 5.0
TARGET_CLIP_PCT = (0.5, 99.5)   # clip training target to these percentiles
RANDOM_RANGE_PCT = (1, 99)      # random baseline draws uniformly in this percentile range

# ---- Model ----
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 128
DROPOUT = 0.1

# ---- Training ----
SEED = 42
BATCH_SIZE = 1024
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 15
SAMPLES_PER_EPOCH = 500_000     # ~3.5 min per epoch on an M1 GPU
PATIENCE = 3
GRAD_CLIP = 1.0

# ---- Baselines ----
ARIMA_ORDER = (1, 0, 1)
ARIMA_WORKERS = 6
