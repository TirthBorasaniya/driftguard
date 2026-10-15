"""Typed runtime settings loaded from environment variables with pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============= Constants =============

REPO_ROOT = Path(__file__).resolve().parents[3]


# ============= Settings =============


class Settings(BaseSettings):
    """
    Environment-driven settings shared by every DriftGuard service.

    Secrets (`admin_token`, `ip_hash_salt`) are `SecretStr` so they never appear in reprs or logs.
    Everything that is not a secret and not deployment-specific lives in YAML under `configs/`.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = Field(default="driftguard", alias="SERVICE_NAME")
    dataset: str = Field(default="cicids2017", alias="DATASET")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    kafka_bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    schema_registry_url: str = Field(default="http://localhost:8081", alias="SCHEMA_REGISTRY_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    prefect_api_url: str = Field(default="http://localhost:4200/api", alias="PREFECT_API_URL")
    serving_url: str = Field(default="http://localhost:8001", alias="SERVING_URL")
    prometheus_url: str = Field(default="http://localhost:9090", alias="PROMETHEUS_URL")

    admin_token: SecretStr = Field(default=SecretStr(""), alias="ADMIN_TOKEN")
    ip_hash_salt: SecretStr = Field(default=SecretStr(""), alias="IP_HASH_SALT")

    config_dir: Path = Field(default=REPO_ROOT / "configs", alias="CONFIG_DIR")
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="DATA_DIR")
    results_dir: Path = Field(default=REPO_ROOT / "results", alias="RESULTS_DIR")
    offline_store_dir: Path = Field(default=REPO_ROOT / "data" / "offline", alias="OFFLINE_DIR")
    feast_repo_path: Path = Field(
        default=REPO_ROOT / "src" / "driftguard" / "features" / "feature_repo",
        alias="FEAST_REPO_PATH",
    )
    orchestrator_db_path: Path = Field(
        default=REPO_ROOT / "data" / "orchestrator.db", alias="ORCHESTRATOR_DB_PATH"
    )
    promotions_log_path: Path = Field(
        default=REPO_ROOT / "results" / "promotions.jsonl", alias="PROMOTIONS_LOG_PATH"
    )
    metrics_port: int = Field(default=9101, alias="METRICS_PORT")
    instance_id: str = Field(default="", alias="INSTANCE_ID")

    def config_path(self, *parts: str) -> Path:
        """Return a path under the configuration directory."""
        return self.config_dir.joinpath(*parts)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance; skip an unreadable `.env` file."""
    try:
        return Settings()
    except PermissionError:
        return Settings(_env_file=None)
