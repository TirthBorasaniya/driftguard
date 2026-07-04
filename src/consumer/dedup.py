"""Idempotent consumer guard: Redis-backed dedup keyed on event_id."""

from redis import Redis

DEDUP_KEY_PREFIX = "processed_event:"
DEDUP_TTL_SECONDS = 3600  # matches the 1-hour Feast feature TTL


def is_duplicate_event(
    redis_client: Redis,
    event_id: str,
    dedup_key_prefix: str = DEDUP_KEY_PREFIX,
    dedup_ttl_seconds: int = DEDUP_TTL_SECONDS,
) -> bool:
    """
    Check and mark an event as processed using Redis SETNX for atomicity.

    Parameters
    ----------
    redis_client : Redis
        Active Redis connection.
    event_id : str
        Unique event identifier.
    dedup_key_prefix : str
        Prefix for the dedup key namespace.
    dedup_ttl_seconds : int
        Time-to-live for the dedup marker.

    Returns
    -------
    o_is_duplicate : bool
        True if the event was already processed, False if this call marked
        it as processed for the first time.
    """
    key = f"{dedup_key_prefix}{event_id}"
    o_is_duplicate = not redis_client.set(key, "1", ex=dedup_ttl_seconds, nx=True)
    return o_is_duplicate
