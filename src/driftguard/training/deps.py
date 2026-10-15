"""Construction of real retraining dependencies from Settings (MLflow, Feast, GX, serving admin)."""

import httpx

from driftguard.common.config import load_config
from driftguard.common.mlflow_client import MlflowRegistry
from driftguard.common.settings import Settings
from driftguard.data.registry import dataset_config, feature_names, stream_features_enabled
from driftguard.features.offline import OfflineStore
from driftguard.training.flow import RetrainDeps
from driftguard.training.validate import GXValidator, load_bounds
from driftguard.training.window import build_training_window

# ============= Constants =============

RESULT_TTL_S = 7 * 24 * 3600


# ============= Builders =============


def model_features(dataset: str) -> list[str]:
    """Model inputs: allowlisted features plus stream features when enabled (SPEC 4.3)."""
    names = feature_names(dataset)
    if stream_features_enabled(dataset):
        from driftguard.streamfeatures.windows import BEHAVIOR_FEATURES

        names = names + list(BEHAVIOR_FEATURES)
    return names


def feature_refs(dataset: str) -> list[str]:
    """Feast feature references for the model inputs."""
    refs = [f"flow_features_{dataset}:{name}" for name in feature_names(dataset)]
    if stream_features_enabled(dataset):
        from driftguard.streamfeatures.windows import BEHAVIOR_FEATURES

        refs += [f"flow_behavior_{dataset}:{name}" for name in BEHAVIOR_FEATURES]
    return refs


def feast_historical(settings: Settings):
    """Return `historical(entity_df, refs)` backed by Feast `get_historical_features`."""
    from driftguard.features.store import build_store, redis_online_store

    store = build_store(
        settings.feast_repo_path,
        settings.data_dir / "feast" / "registry.db",
        redis_online_store(settings.redis_url),
    )

    def historical(entity_df, refs):
        """Point-in-time join through Feast `get_historical_features`."""
        return store.get_historical_features(entity_df=entity_df, features=refs).to_df()

    return historical


def rollout_setter(settings: Settings):
    """Return `set_mode(dataset, mode)` calling the token-protected serving admin API."""

    def set_mode(dataset: str, mode: str) -> None:
        """Set the rollout mode through the serving admin API."""
        response = httpx.post(
            f"{settings.serving_url}/admin/rollout",
            json={"dataset": dataset, "mode": mode},
            headers={"Authorization": f"Bearer {settings.admin_token.get_secret_value()}"},
            timeout=10.0,
        )
        response.raise_for_status()

    return set_mode


def status_reporter(settings: Settings):
    """Return `report(trigger_id, status)` writing the flow's final status for the orchestrator."""
    from redis import Redis

    from driftguard.orchestrator.triggers import RESULT_KEY

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return lambda trigger_id, status: redis.set(
        RESULT_KEY.format(trigger_id=trigger_id), status, ex=RESULT_TTL_S
    )


def build_retrain_deps(dataset: str, settings: Settings) -> RetrainDeps:
    """Assemble every dependency of a retraining run."""
    offline = OfflineStore(settings.offline_store_dir)
    historical = feast_historical(settings)
    refs = feature_refs(dataset)
    gx_cfg = dataset_config(dataset)["gx"]
    validator = GXValidator(
        load_bounds(settings.config_path("data", f"{dataset}_gx_bounds.json")),
        float(gx_cfg["mostly"]),
        int(gx_cfg["min_training_rows"]),
        name=f"{dataset}_training",
    )
    return RetrainDeps(
        registry=MlflowRegistry(settings.mlflow_tracking_uri),
        build_window=lambda end_ms, n: build_training_window(
            offline, dataset, end_ms, n, refs, historical
        ),
        validator=validator,
        set_rollout_mode=rollout_setter(settings),
        report_status=status_reporter(settings),
        model_config=load_config(settings.config_path("model.yaml")).data,
        retrain_config=load_config(settings.config_path("retrain.yaml")).data,
        features=model_features(dataset),
    )
