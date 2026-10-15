"""Feast PushSource and feature view for behavioral features, and the consumer's behavior sink."""

from collections.abc import Callable
from datetime import timedelta

import pandas as pd
from feast import FeatureView, Field, FileSource, PushSource
from feast.types import Float32

from driftguard.features.offline import OfflineStore
from driftguard.streamfeatures.windows import BEHAVIOR_FEATURES

# ============= Constants =============

ONLINE_TTL = timedelta(days=2)


def push_source_name(dataset: str) -> str:
    """Name of the PushSource for a dataset."""
    return f"flow_behavior_{dataset}_push"


def view_name(dataset: str) -> str:
    """Name of the behavioral feature view for a dataset."""
    return f"flow_behavior_{dataset}"


# ============= Feast objects =============


def behavior_objects(dataset: str, offline: OfflineStore) -> list:
    """
    PushSource-backed feature view `flow_behavior_<dataset>`.

    The batch source is the date-partitioned `flow_behavior` table that the consumer appends to,
    so `get_historical_features` returns exactly the values that were served.
    """
    from driftguard.features.store import EVENT_ENTITY

    batch = FileSource(
        name=f"flow_behavior_{dataset}_source",
        path=str(offline.table_path(dataset, "flow_behavior")),
        timestamp_field="event_timestamp",
    )
    push = PushSource(name=push_source_name(dataset), batch_source=batch)
    view = FeatureView(
        name=view_name(dataset),
        entities=[EVENT_ENTITY],
        ttl=ONLINE_TTL,
        schema=[Field(name=name, dtype=Float32) for name in BEHAVIOR_FEATURES],
        source=push,
        online=True,
    )
    return [push, view]


# ============= Sink =============


class BehaviorSink:
    """
    Writes each batch's behavioral features online through the PushSource and offline to parquet.

    Parameters
    ----------
    dataset : str
        Dataset name.
    offline : OfflineStore
        Offline store (date-partitioned `flow_behavior` table).
    push_online : callable or None
        `push_online(push_source_name, frame)`, normally a wrapper over
        `FeatureStore.push(..., to=PushMode.ONLINE)`.
    """

    def __init__(
        self,
        dataset: str,
        offline: OfflineStore | None,
        push_online: Callable[[str, pd.DataFrame], None] | None,
    ):
        self.dataset = dataset
        self.offline = offline
        self.push_online = push_online

    def write(self, frame: pd.DataFrame) -> None:
        """Persist one batch (`event_id`, `event_timestamp`, behavioral columns as float32)."""
        if frame.empty:
            return
        if self.push_online is not None:
            self.push_online(push_source_name(self.dataset), frame)
        if self.offline is not None:
            self.offline.append(self.dataset, "flow_behavior", frame)


def feast_online_pusher(store) -> Callable[[str, pd.DataFrame], None]:
    """Return a callable pushing a frame to the online store through a PushSource."""
    from feast.data_source import PushMode

    return lambda name, frame: store.push(name, frame, to=PushMode.ONLINE)
