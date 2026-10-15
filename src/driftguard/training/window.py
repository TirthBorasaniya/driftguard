"""Training window construction with point-in-time features and time-ordered splits (SPEC 6.2)."""

import hashlib
from dataclasses import dataclass

import pandas as pd

from driftguard.features.offline import OfflineStore

# ============= Constants =============

META = ["event_id", "event_timestamp", "label", "label_family"]


# ============= Window =============


def labeled_events(offline: OfflineStore, dataset: str, window_end_ms: int) -> pd.DataFrame:
    """
    Join delayed labels to served events, keeping events at or before `window_end_ms`.

    Labels are de-duplicated by `event_id` (the label sink is at-least-once). Only labels that
    had arrived by the window end (`label_time <= window_end_ms`) are used, which is what the
    system knew at trigger time.

    Returns
    -------
    events : pd.DataFrame
        Columns `event_id`, `event_timestamp`, `label`, `label_family`, sorted by event time.
    """
    labels = offline.read(dataset, "labels", ["event_id", "label", "label_family", "label_time"])
    if labels.empty:
        return pd.DataFrame(columns=META)
    labels = labels[labels["label_time"] <= window_end_ms].drop_duplicates("event_id", keep="last")
    served = offline.read(dataset, "flow_features", ["event_id", "event_timestamp"])
    served = served.drop_duplicates("event_id", keep="first")
    events = served.merge(labels[["event_id", "label", "label_family"]], on="event_id")
    end = pd.Timestamp(window_end_ms, unit="ms", tz="UTC")
    events = events[events["event_timestamp"] <= end]
    return events.sort_values(["event_timestamp", "event_id"], kind="mergesort").reset_index(
        drop=True
    )


def attach_features(
    events: pd.DataFrame,
    feature_refs: list[str],
    historical: callable,
) -> pd.DataFrame:
    """
    Attach features with a point-in-time join.

    Parameters
    ----------
    events : pd.DataFrame
        Entity rows (`event_id`, `event_timestamp`) plus labels.
    feature_refs : list of str
        Feast references, for example `flow_features_cicids2017:flow_duration`.
    historical : callable
        `historical(entity_df, feature_refs) -> DataFrame`, normally
        `FeatureStore.get_historical_features(...).to_df()`.

    Returns
    -------
    frame : pd.DataFrame
        Events with feature columns, in event-time order.
    """
    joined = historical(events[["event_id", "event_timestamp"]], feature_refs)
    names = [ref.split(":", 1)[1] for ref in feature_refs]
    merged = events.merge(joined[["event_id", *names]], on="event_id", how="left")
    if merged[names].isna().any().any():
        missing = int(merged[names].isna().any(axis=1).sum())
        raise ValueError(f"{missing} labeled events have no features in the offline store")
    return merged.sort_values(["event_timestamp", "event_id"], kind="mergesort").reset_index(
        drop=True
    )


def build_training_window(
    offline: OfflineStore,
    dataset: str,
    window_end_ms: int,
    n_labeled: int,
    feature_refs: list[str],
    historical: callable,
) -> pd.DataFrame:
    """Most recent `n_labeled` labeled events up to `window_end_ms`, with point-in-time features."""
    events = labeled_events(offline, dataset, window_end_ms).tail(n_labeled)
    if events.empty:
        raise ValueError("no labeled events available for the training window")
    return attach_features(events, feature_refs, historical)


def direct_historical(offline: OfflineStore, dataset: str):
    """
    Point-in-time join read straight from the offline parquet (tests and offline tools).

    Each event's own feature row is keyed by `event_id` and stamped with its event time, so the
    exact-key join is point-in-time correct.
    """

    def historical(entity_df: pd.DataFrame, feature_refs: list[str]) -> pd.DataFrame:
        names = [ref.split(":", 1)[1] for ref in feature_refs]
        stored = offline.read(dataset, "flow_features", ["event_id", "event_timestamp", *names])
        stored = stored.drop_duplicates("event_id", keep="first")
        merged = entity_df.merge(stored, on="event_id", suffixes=("", "_stored"))
        merged = merged[merged["event_timestamp_stored"] <= merged["event_timestamp"]]
        return merged[["event_id", *names]]

    return historical


# ============= Splits =============


@dataclass
class WindowSplits:
    """Time-ordered splits of a training window."""

    fit: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame


def split_window(
    frame: pd.DataFrame, holdout_fraction: float, validation_fraction: float
) -> WindowSplits:
    """
    Split by event time: the newest `holdout_fraction` is held out (never used for fitting or
    threshold selection); the validation slice is the end of the remaining training portion.
    """
    n = len(frame)
    holdout_start = int(round(n * (1 - holdout_fraction)))
    training = frame.iloc[:holdout_start]
    val_start = int(round(len(training) * (1 - validation_fraction)))
    return WindowSplits(
        fit=training.iloc[:val_start].reset_index(drop=True),
        validation=training.iloc[val_start:].reset_index(drop=True),
        holdout=frame.iloc[holdout_start:].reset_index(drop=True),
    )


def data_hash(frame: pd.DataFrame) -> str:
    """SHA-256 over the sorted event ids of a window (identifies the exact training data)."""
    hasher = hashlib.sha256()
    for event_id in sorted(frame["event_id"].astype(str)):
        hasher.update(event_id.encode())
    return hasher.hexdigest()
