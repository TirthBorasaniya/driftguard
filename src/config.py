"""Single source of truth: environment settings, paths, feature definitions, and hyperparameters."""

from pathlib import Path

from pydantic_settings import BaseSettings


# ============= Environment Settings =============


class Settings(BaseSettings):
    """All runtime configuration, readable from environment variables or .env file."""

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "transactions"
    kafka_dlq_topic: str = "transactions.dlq"
    kafka_group_id: str = "fraud-consumer"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment_name: str = "fraud-detection"
    mlflow_model_name: str = "fraud-detector"
    mlflow_champion_alias: str = "champion"
    mlflow_challenger_alias: str = "challenger"

    # Model
    scale_pos_weight: float = 172.0
    champion_improvement_threshold: float = 0.01

    # Drift detection
    drift_min_window: int = 500
    drift_share_threshold: float = 0.3

    # Self-healing
    healing_mode: str = "AUTO"
    healing_cooldown_hours: int = 6

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: str = "http://localhost:3000"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()


# ============= Paths =============

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ENCODERS_DIR = DATA_DIR / "encoders"
DRIFT_SCENARIOS_DIR = DATA_DIR / "drift_scenarios"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_REPO_DIR = PROJECT_ROOT / "src" / "features" / "feature_repo"

# raw data files
TRAIN_CSV = RAW_DIR / "fraudTrain.csv"
TEST_CSV = RAW_DIR / "fraudTest.csv"

# processed splits
TRAIN_FILE = PROCESSED_DIR / "train.parquet"
TEST_FILE = PROCESSED_DIR / "test.parquet"
REFERENCE_FILE = PROCESSED_DIR / "reference.parquet"
STREAM_FILE = PROCESSED_DIR / "stream.parquet"

# aggregated feature tables for Feast
CARD_STATS_FILE = PROCESSED_DIR / "card_stats_7d.parquet"
CATEGORY_STATS_FILE = PROCESSED_DIR / "category_fraud_rate.parquet"

# model artifacts
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
CHAMPION_METRICS_PATH = MODELS_DIR / "champion_metrics.json"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.json"

# database
DB_PATH = PROJECT_ROOT / "predictions.db"


# ============= Feature Definitions =============

TARGET_COL = "is_fraud"
ENTITY_COL = "cc_num"
TIME_COL = "trans_date_trans_time"

# categorical columns — encoded with SafeLabelEncoder, cat_indices passed to LightGBM
CATEGORICAL_COLS = ["merchant", "category", "gender", "city", "state", "zip", "job"]

# numeric columns — raw + derived
NUMERIC_COLS = [
    "amt",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "hour_of_day",
    "day_of_week",
    "age",
    "distance_km",
    "amt_log",
    # feast rolling features
    "txn_count_7d",
    "amt_mean_7d",
    "amt_max_7d",
    "txn_velocity_7d",
    "category_fraud_rate",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


# ============= LightGBM Hyperparameters =============

LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 3000,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "scale_pos_weight": settings.scale_pos_weight,
}

EARLY_STOPPING_ROUNDS = 50
LOG_EVALUATION_PERIOD = 100
