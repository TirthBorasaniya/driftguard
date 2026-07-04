"""Prefect flow: point-in-time correct offline Feast materialization."""

from prefect import flow, task
from prefect.logging import get_run_logger

from src.config import FEATURE_REPO_DIR
from src.features.materializer import MATERIALIZATION_LOOKBACK_HOURS, materialize_offline_features


@task(name="materialize-offline-features")
def materialize_offline_features_task(lookback_hours: int):
    """Run point-in-time correct offline materialization, distinct from the online write path."""
    logger = get_run_logger()
    features_df = materialize_offline_features(str(FEATURE_REPO_DIR), lookback_hours)
    logger.info(f"Offline materialization complete: {len(features_df)} rows")
    return features_df


@flow(name="offline-materialization-flow")
def offline_materialization_flow(lookback_hours: int = MATERIALIZATION_LOOKBACK_HOURS):
    """
    Standalone flow to run point-in-time correct offline Feast materialization.

    Distinct from the consumer's online Redis write path and from
    `materialize_to_online_store`: this pulls a point-in-time correct join
    from the offline store for downstream training/analysis use, without
    touching the online store used at serving time.

    Parameters
    ----------
    lookback_hours : int
        How far back to materialize from the offline store.
    """
    return materialize_offline_features_task(lookback_hours)


if __name__ == "__main__":
    offline_materialization_flow()
