"""Single source of truth: environment settings, paths, feature definitions, and hyperparameters."""

from pathlib import Path

from pydantic_settings import BaseSettings

from src.features.engineering import FEATURE_COLS

# ============= Environment Settings =============


class Settings(BaseSettings):
    """All runtime configuration, readable from environment variables or .env file."""

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "network_flows"
    kafka_dlq_topic: str = "network_flows.dlq"
    kafka_group_id: str = "network-flow-consumer"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment_name: str = "network_anomaly_detection"
    mlflow_model_name: str = "network-anomaly-detector"
    mlflow_champion_alias: str = "champion"
    mlflow_challenger_alias: str = "challenger"

    # Model
    # tune to the observed benign/attack class ratio of the loaded CICIDS2017 window
    scale_pos_weight: float = 10.0
    # challenger must exceed champion PR-AUC by this margin to be promoted
    champion_improvement_threshold: float = 0.005

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
CICIDS_DATA_DIR = RAW_DIR / "cicids2017"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
ENCODERS_DIR = DATA_DIR / "encoders"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURE_REPO_DIR = PROJECT_ROOT / "src" / "features" / "feature_repo"

# raw CICIDS2017 capture files in chronological (replay) order
CICIDS_FILES_ORDERED = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]
# Monday is the benign-only capture day used as the clean drift baseline
REFERENCE_CAPTURE_FILE = CICIDS_DATA_DIR / "Monday-WorkingHours.pcap_ISCX.csv"

# processed splits
TRAIN_FILE = PROCESSED_DIR / "train.parquet"
TEST_FILE = PROCESSED_DIR / "test.parquet"
REFERENCE_FILE = REFERENCE_DIR / "reference_network_flows.parquet"
STREAM_FILE = PROCESSED_DIR / "stream.parquet"

# per-entity aggregated feature table backing the Feast offline source
NETWORK_FLOW_STATS_FILE = PROCESSED_DIR / "network_flow_features.parquet"

# model artifacts
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
CHAMPION_METRICS_PATH = MODELS_DIR / "champion_metrics.json"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.json"

# database
DB_PATH = PROJECT_ROOT / "predictions.db"


# ============= CICIDS2017 Field Mapping =============

# maps raw CICIDS2017 CSV column headers to NetworkFlowEvent schema field names;
# shared by the producer, preprocessing, and the reference dataset generator
CICIDS_COLUMN_MAP = {
    "Flow Duration": "flow_duration",
    "Flow Bytes/s": "flow_bytes_per_sec",
    "Flow Packets/s": "flow_packets_per_sec",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_length_fwd_packets",
    "Total Length of Bwd Packets": "total_length_bwd_packets",
    "Packet Length Mean": "packet_length_mean",
    "Packet Length Std": "packet_length_std",
    "Flow IAT Mean": "flow_iat_mean",
    "SYN Flag Count": "syn_flag_count",
    "Label": "label",
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Source Port": "src_port",
    "Destination Port": "dst_port",
    "Protocol": "protocol",
    "Flow ID": "flow_id",
    "Timestamp": "timestamp_raw",
}

# only BENIGN maps to 0; every other label is an attack and maps to 1
BENIGN_LABEL = "BENIGN"


# ============= Feature Definitions =============

TARGET_COL = "label_binary"
ENTITY_COL = "src_ip"
TIME_COL = "timestamp_utc"

# network flow features are all numeric; there are no categorical model inputs
CATEGORICAL_COLS: list[str] = []

# numeric columns equal the canonical FEATURE_COLS for the network anomaly model
NUMERIC_COLS = list(FEATURE_COLS)


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
