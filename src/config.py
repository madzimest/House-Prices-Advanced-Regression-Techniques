# src/config.py
from pathlib import Path

# Project root (assuming src/ is inside DS/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"
OOF_DIR = PROJECT_ROOT / "models" / "oof"

# Files
TRAIN_FILE = DATA_RAW / "train.csv"
TEST_FILE = DATA_RAW / "test.csv"
SUBMISSION_FILE = PROJECT_ROOT / "notebooks" / "submission.csv"

# Modeling settings
RANDOM_STATE = 42
N_SPLITS = 10
TARGET = "SalePrice"
LOG_TARGET = True           # use log(1+SalePrice)

# Feature engineering
SKEW_THRESHOLD = 0.75

# LightGBM defaults
LGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Ridge defaults
RIDGE_ALPHA = 10.0

# Ensemble blending weights (LGB, Ridge)
BLEND_WEIGHTS = [0.6, 0.4]