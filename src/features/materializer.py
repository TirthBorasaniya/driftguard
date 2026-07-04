"""Materialize offline Feast features to the Redis online store."""

import subprocess
import sys

from src.config import FEATURE_REPO_DIR

MATERIALIZATION_LOOKBACK_HOURS = 24


def apply_feast() -> bool:
    """
    Run feast apply to register feature definitions.

    Returns
    -------
    success : bool
    """
    result = subprocess.run(
        [sys.executable, "-m", "feast", "apply"],
        cwd=str(FEATURE_REPO_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"feast apply failed:\n{result.stderr}")
    print(result.stdout)
    return True


def materialize_to_online_store() -> bool:
    """
    Push all feature views to the Redis online store.

    Returns
    -------
    success : bool
    """
    from datetime import datetime

    end_date = datetime.now().isoformat()
    result = subprocess.run(
        [sys.executable, "-m", "feast", "materialize-incremental", end_date],
        cwd=str(FEATURE_REPO_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"feast materialize failed:\n{result.stderr}")
    print(result.stdout)
    return True


def materialize_offline_features(
    feast_repo_path: str,
    lookback_hours: int = MATERIALIZATION_LOOKBACK_HOURS,
):
    """
    Run point-in-time correct offline materialization from the Feast offline
    store, separate from the online write path used at serving time.

    Reads the per-src_ip offline feature table, builds an entity dataframe
    from the most recent event per src_ip observed within the lookback
    window, and joins it against the offline store via
    FeatureStore.get_historical_features (point-in-time correct).

    Parameters
    ----------
    feast_repo_path : str
        Path to the Feast feature repo.
    lookback_hours : int
        How far back to materialize from the offline store.

    Returns
    -------
    features_df : pd.DataFrame
        Point-in-time correct feature values for each src_ip observed within
        the lookback window.
    """
    from datetime import datetime, timedelta, timezone

    import pandas as pd
    from feast import FeatureStore

    from src.config import FEATURE_COLS, NETWORK_FLOW_STATS_FILE

    source_df = pd.read_parquet(NETWORK_FLOW_STATS_FILE)
    source_df["event_timestamp"] = pd.to_datetime(source_df["event_timestamp"], utc=True)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    window_df = source_df[source_df["event_timestamp"] >= cutoff]

    entity_df = window_df.groupby("src_ip", as_index=False)["event_timestamp"].max()

    store = FeatureStore(repo_path=feast_repo_path)
    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=[f"network_flow_features:{col}" for col in FEATURE_COLS],
    ).to_df()

    return features_df


if __name__ == "__main__":
    apply_feast()
    materialize_to_online_store()
