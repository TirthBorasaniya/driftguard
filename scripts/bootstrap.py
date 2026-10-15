"""Prepare a stack: topics, Avro subjects, Feast definitions, and the initial champions."""

import argparse
import sys

from driftguard.common.config import load_config
from driftguard.common.kafka import ensure_topics
from driftguard.common.logging import configure_logging
from driftguard.common.mlflow_client import CHAMPION, MlflowRegistry, model_name
from driftguard.common.promotions import append_promotion
from driftguard.common.settings import get_settings
from driftguard.data.registry import stream_features_enabled
from driftguard.data.schemas import subject_name
from driftguard.data.serde import register_subject, registry_client
from driftguard.features.offline import OfflineStore
from driftguard.features.store import build_store, definitions, redis_online_store
from driftguard.training.deps import model_features
from driftguard.training.flow import compare_on_holdout, train_and_package
from driftguard.training.window import data_hash, split_window

# ============= Initial data =============


def initial_frame(dataset: str, source: str, bootstrap_cfg: dict):
    """Rows the initial champion trains on (real data, or synthetic rows for CI)."""
    if source == "synthetic":
        from driftguard.data.synthetic import synthetic_flows

        return synthetic_flows(dataset, 20_000, attack_rate=0.1, seed=7)
    if dataset == "cicids2017":
        from driftguard.data.cicids2017 import load_dataset

        frame, _ = load_dataset(days=list(bootstrap_cfg["cicids2017"]["train_days"]))
        return frame
    from driftguard.data.kyoto import load_years

    return load_years(list(bootstrap_cfg["kyoto"]["train_years"]))


def bootstrap_champion(dataset: str, source: str, settings, o_force: bool) -> str | None:
    """
    Train and register the initial champion unless one exists.

    Parameters
    ----------
    dataset : str
        Dataset name.
    source : str
        `dataset` or `synthetic`.
    settings : object
        Runtime settings.
    o_force : bool
        Replace an existing champion.

    Returns
    -------
    version : str or None
        The registered version, or None when a champion already existed.
    """
    registry = MlflowRegistry(settings.mlflow_tracking_uri)
    name = model_name(dataset)
    existing = registry.get_alias(name, CHAMPION)
    if existing is not None and not o_force:
        print(f"{name}@champion exists (version {existing.version}); skipping")
        return None
    bootstrap_cfg = load_config(settings.config_path("bootstrap.yaml")).data
    model_cfg = load_config(settings.config_path("model.yaml")).data
    frame = initial_frame(dataset, source, bootstrap_cfg)
    features = model_features(dataset)
    if stream_features_enabled(dataset):
        # same code path as the consumer, so the champion trains on values it will be served
        from driftguard.streamfeatures.offline import redis_replay_features

        frame = redis_replay_features(frame)
    frame = frame.sort_values(["event_time", "event_id"], kind="mergesort").reset_index(drop=True)
    splits = split_window(
        frame, float(bootstrap_cfg["holdout_fraction"]), float(bootstrap_cfg["validation_fraction"])
    )
    trained = train_and_package(
        splits, features, model_cfg, int(bootstrap_cfg["reference_max_rows"])
    )
    holdout = compare_on_holdout(splits.holdout, trained, None, None)["challenger"]
    tags = {
        "trigger_id": "bootstrap",
        "window_end": int(frame["event_time"].max()),
        "train_rows": len(splits.fit) + len(splits.validation),
        "holdout_rows": len(splits.holdout),
        "data_hash": data_hash(frame),
        "threshold": trained.threshold,
        "parity_max_abs_diff": trained.parity_max_abs_diff,
        "source": source,
    }
    metrics = {
        f"holdout_{k}": holdout[k]
        for k in ("pr_auc", "roc_auc", "precision", "recall")
        if holdout[k] is not None
    }
    version = registry.register(dataset, trained.artifacts(), tags, metrics, trained.params)
    registry.set_alias(name, CHAMPION, version)
    append_promotion(
        settings.promotions_log_path,
        {
            "dataset": dataset,
            "action": "bootstrap",
            "champion": version,
            "previous_champion": None,
            "reason": "initial champion (no incumbent to gate against)",
        },
    )
    print(f"Registered {name} version {version} as champion")
    return version


# ============= CLI =============


def main() -> int:
    """Run every bootstrap step that is not already done."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--source", choices=["dataset", "synthetic"], default="dataset")
    parser.add_argument("--o-force-champion", action="store_true", dest="o_force")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging("bootstrap", settings.log_level)
    datasets = tuple(args.datasets or [settings.dataset])

    ensure_topics(settings.kafka_bootstrap_servers, datasets)
    client = registry_client(settings.schema_registry_url)
    for subject in [subject_name("labels"), subject_name("predictions"), subject_name("drift")]:
        register_subject(client, subject)
    offline = OfflineStore(settings.offline_store_dir)
    store = build_store(
        settings.feast_repo_path,
        settings.data_dir / "feast" / "registry.db",
        redis_online_store(settings.redis_url),
    )
    for dataset in datasets:
        register_subject(client, subject_name("flows", dataset))
        store.apply(definitions(dataset, offline))
        bootstrap_champion(dataset, args.source, settings, args.o_force)
        from redis import Redis

        Redis.from_url(settings.redis_url).set(f"dg:rollout:{dataset}", "champion_only", nx=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
