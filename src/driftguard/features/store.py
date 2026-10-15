"""Programmatic Feast repository: entity, batch sources, and feature views per dataset."""

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.repo_config import RepoConfig
from feast.types import Float32

from driftguard.data.registry import feature_names
from driftguard.features.offline import OfflineStore

# ============= Constants =============

PROJECT = "driftguard"
ONLINE_TTL = timedelta(days=2)
EVENT_ENTITY = Entity(name="event", join_keys=["event_id"], value_type=ValueType.STRING)


# ============= Store construction =============


def build_store(
    repo_path: Path,
    registry_path: Path,
    online_store: dict,
) -> FeatureStore:
    """
    Build a FeatureStore from code instead of a YAML file so connection details come from Settings.

    Parameters
    ----------
    repo_path : Path
        Working directory for Feast.
    registry_path : Path
        SQLite registry file.
    online_store : dict
        Online store config, for example `{"type": "redis", "connection_string": "redis:6379",
        "key_ttl_seconds": 172800}`.

    Returns
    -------
    store : FeatureStore
    """
    repo_path.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    config = RepoConfig(
        project=PROJECT,
        registry=str(registry_path),
        provider="local",
        online_store=online_store,
        offline_store={"type": "file"},
        entity_key_serialization_version=3,
        repo_path=str(repo_path),
    )
    return FeatureStore(config=config)


def redis_online_store(redis_url: str, ttl_s: int = 172_800) -> dict:
    """Translate `redis://host:port/db` into a Feast Redis online store config."""
    hostport = redis_url.split("://", 1)[-1].split("/", 1)[0]
    return {"type": "redis", "connection_string": hostport, "key_ttl_seconds": ttl_s}


def flow_features_view(dataset: str, offline: OfflineStore) -> FeatureView:
    """Per-event allowlisted features, keyed by `event_id`, backed by the partitioned parquet."""
    source = FileSource(
        name=f"flow_features_{dataset}_source",
        path=str(offline.table_path(dataset, "flow_features")),
        timestamp_field="event_timestamp",
    )
    return FeatureView(
        name=f"flow_features_{dataset}",
        entities=[EVENT_ENTITY],
        ttl=ONLINE_TTL,
        schema=[Field(name=name, dtype=Float32) for name in feature_names(dataset)],
        source=source,
        online=True,
    )


def definitions(dataset: str, offline: OfflineStore) -> list:
    """Return every Feast object for a dataset (stream view added when enabled)."""
    objects = [EVENT_ENTITY, flow_features_view(dataset, offline)]
    from driftguard.data.registry import stream_features_enabled

    if stream_features_enabled(dataset):
        from driftguard.streamfeatures.push import behavior_objects

        objects += behavior_objects(dataset, offline)
    return objects
