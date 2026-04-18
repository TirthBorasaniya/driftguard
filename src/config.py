"""Single source of truth for paths, thresholds, hyperparameters, and feature definitions."""

import os
from pathlib import Path

# ============= Project Paths =============

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
BATCH_DIR = DATA_DIR / "production_batches"
FEATURES_DIR = DATA_DIR / "features"
ENCODERS_DIR = DATA_DIR / "encoders"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEAST_DIR = PROJECT_ROOT / "feast"

# ============= Raw Data Files =============

RAW_TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
RAW_IDENTITY_FILE = RAW_DIR / "train_identity.csv"

# ============= Processed Data Files =============

TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
REFERENCE_FILE = REFERENCE_DIR / "reference.csv"
BATCH_FILES = [BATCH_DIR / f"batch_{i}.csv" for i in range(3)]

# ============= Model Artifacts =============

PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.json"
CATEGORICAL_COLS_PATH = MODELS_DIR / "categorical_cols.json"
METRICS_HISTORY_PATH = MODELS_DIR / "metrics_history.csv"

# ============= MLflow Configuration =============

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'mlflow.db'}",
)
MLFLOW_EXPERIMENT_NAME = "ieee-fraud-detection"
MLFLOW_MODEL_NAME = "fraud-detector-ieee"
MLFLOW_CHAMPION_ALIAS = "champion"
MLFLOW_CHALLENGER_ALIAS = "challenger"

# ============= Feature Definitions =============

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

CATEGORICAL_COLS = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo",
    "id_12", "id_15", "id_16", "id_28",
]

D_COLS_TO_NORMALIZE = ["D1", "D2", "D3", "D4", "D5", "D10", "D15"]

V_COLS = [
    "V1", "V3", "V4", "V6", "V8", "V11", "V13", "V14", "V17", "V20",
    "V23", "V26", "V27", "V30", "V36", "V37", "V40", "V44", "V47", "V54",
    "V56", "V59", "V62", "V67", "V75", "V78", "V80", "V82", "V86", "V87",
    "V126", "V127", "V128", "V130",
]

NUMERIC_COLS = (
    # engineered features (3)
    ["TransactionAmt_log", "transaction_day", "transaction_hour"]
    # normalized D columns (7)
    + [f"{col}n" for col in D_COLS_TO_NORMALIZE]
    # raw numeric from transaction table (8)
    + ["card1", "card2", "card3", "card5"]
    + ["addr1", "addr2"]
    + ["dist1", "dist2"]
    # counting features (14)
    + [f"C{i}" for i in range(1, 15)]
    # identity numeric (11)
    + [f"id_{i:02d}" for i in range(1, 12)]
    # selected V columns for predictive power (34)
    + V_COLS
)

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# ============= Data Splitting =============

SPLIT_RATIOS = {
    "train": 0.70,
    "test": 0.10,
    "reference": 0.07,
    "batch_0": 0.04,
    "batch_1": 0.04,
    "batch_2": 0.05,
}

# ============= LightGBM Hyperparameters =============

LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "num_leaves": 256,
    "learning_rate": 0.01,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 5000,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "is_unbalance": False,
}

SMOTE_RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
LOG_EVALUATION_PERIOD = 100

# ============= Drift Detection Thresholds =============

DRIFT_SHARE_THRESHOLD = 0.5
DRIFT_DATASET_THRESHOLD = 0.05
PREDICTION_DRIFT_THRESHOLD = 0.1

# ============= Self-Healing Configuration =============

HEALING_MODE = os.getenv("HEALING_MODE", "AUTO")
HEALING_COOLDOWN_HOURS = int(os.getenv("HEALING_COOLDOWN_HOURS", "6"))
CHAMPION_IMPROVEMENT_THRESHOLD = 0.005

# ============= API Configuration =============

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
DB_PATH = PROJECT_ROOT / "predictions.db"

# ============= Prometheus Metrics =============

METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
