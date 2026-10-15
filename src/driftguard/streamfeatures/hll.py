"""HyperLogLog bucket keys and multi-bucket unique counts (standard error about 0.81 percent)."""

# ============= Buckets =============


def bucket_of(event_time_ms: int, bucket_s: int) -> int:
    """Index of the `bucket_s` event-time bucket containing an event."""
    return int(event_time_ms) // (bucket_s * 1000)


def bucket_range(last_bucket: int, n_buckets: int) -> list[int]:
    """The `n_buckets` buckets ending at `last_bucket`, oldest first."""
    return list(range(last_bucket - n_buckets + 1, last_bucket + 1))


def hll_key(prefix: str, kind: str, host_key: str, bucket: int) -> str:
    """Key of one HyperLogLog bucket, for example `dg:sf:hll:ports:<src>:<bucket>`."""
    return f"{prefix}hll:{kind}:{host_key}:{bucket}"


def merged_count(pipe, keys: list[str]) -> None:
    """
    Queue the union cardinality of several HyperLogLog buckets.

    `PFCOUNT k1 .. kn` merges the buckets server side into a temporary HyperLogLog and counts it,
    which is the PFMERGE-then-PFCOUNT of SPEC 4.3 without writing a destination key.
    """
    pipe.pfcount(*keys)
