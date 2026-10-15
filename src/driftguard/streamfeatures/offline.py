"""Replay a dataset frame through the stream-feature computer (E1 ablation, initial champion)."""

import uuid

import numpy as np
import pandas as pd

from driftguard.common.config import load_config
from driftguard.common.settings import get_settings
from driftguard.streamfeatures.keys import HostKeyer
from driftguard.streamfeatures.windows import BEHAVIOR_FEATURES, StreamFeatureComputer

# ============= Replay =============


def replay_features(
    frame: pd.DataFrame, computer: StreamFeatureComputer, batch_size: int
) -> pd.DataFrame:
    """
    Compute behavioral features for a frame in event-time order with consumer-sized batches.

    Parameters
    ----------
    frame : pd.DataFrame
        Adapter rows with `event_id`, `event_time`, `src_ip`, `dst_ip`, and source fields.
    computer : StreamFeatureComputer
        Computer with fresh state.
    batch_size : int
        Rows per batch (the consumer's poll size).

    Returns
    -------
    enriched : pd.DataFrame
        The input rows (original order) with BEHAVIOR_FEATURES columns (float32) added.
    """
    order = np.lexsort((frame["event_id"].astype(str).to_numpy(), frame["event_time"].to_numpy()))
    needed = [
        "event_id",
        "event_time",
        "src_ip",
        "dst_ip",
        computer.port_field,
        computer.bytes_field,
        computer.syn_field,
    ]
    records = frame.iloc[order][needed].to_dict("records")
    values = np.empty((len(frame), len(BEHAVIOR_FEATURES)), dtype=np.float32)
    for start in range(0, len(records), batch_size):
        batch = computer.compute(records[start : start + batch_size])
        values[order[start : start + len(batch)]] = [
            [row[name] for name in BEHAVIOR_FEATURES] for row in batch
        ]
    enriched = frame.copy()
    for j, name in enumerate(BEHAVIOR_FEATURES):
        enriched[name] = values[:, j]
    return enriched


def redis_replay_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Replay against Redis under a private key prefix, then delete that state.

    Uses the same computer and batch size as the consumer. Values match the served ones up to the
    composition of poll batches (unique counts and means are computed per batch); training on the
    offline store uses the exact served values.
    """
    from redis import Redis

    settings = get_settings()
    config = dict(load_config(settings.config_path("stream_features.yaml")).data)
    config["key_prefix"] = f"{config['key_prefix']}replay:{uuid.uuid4().hex}:"
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    computer = StreamFeatureComputer(
        redis_client, HostKeyer(settings.ip_hash_salt.get_secret_value()), config
    )
    try:
        return replay_features(frame, computer, int(config["replay_batch_size"]))
    finally:
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(
                cursor=cursor, match=f"{config['key_prefix']}*", count=10_000
            )
            if keys:
                redis_client.delete(*keys)
            if cursor == 0:
                break
