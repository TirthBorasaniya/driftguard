"""Behavioral features over 10 s and 60 s event-time windows, computed in Redis per batch."""

from typing import Any

from driftguard.common.config import load_config
from driftguard.common.metrics import LATE_EVENTS
from driftguard.streamfeatures.hll import bucket_of, bucket_range, hll_key, merged_count
from driftguard.streamfeatures.keys import HostKeyer

# ============= Constants =============

BEHAVIOR_FEATURES = (
    "src_conn_count_10s",
    "src_conn_count_60s",
    "dst_conn_count_10s",
    "src_unique_dst_ports_60s",
    "src_unique_dst_hosts_60s",
    "src_mean_fwd_bytes_60s",
    "src_syn_flow_share_60s",
)


# ============= Computer =============


class StreamFeatureComputer:
    """
    Computes SPEC 4.3 behavioral features for poll batches.

    State per source key: a sorted set of event ids scored by event time (10 s and 60 s counts),
    HyperLogLogs of destination ports and destination hosts per 10 s bucket, and a hash of
    forward-byte sum, flow count, and SYN-flow count per bucket. Per destination key: a sorted set
    (10 s count). Every key expires after twice its window.

    Counts use `ZCOUNT (t - w, t]`, so they include the current event and exclude events with a
    later event time that were processed earlier. Unique counts, mean bytes, and SYN share are
    computed once per source key per batch over the six buckets ending at that key's newest
    bucket in the batch. All writes run in one pipeline and all reads in a second one.

    Parameters
    ----------
    redis_client : redis.Redis
        Client with `decode_responses=True`.
    keyer : HostKeyer
        Salted host keys.
    config : dict
        `configs/stream_features.yaml` content.
    dataset : str
        Dataset label for metrics.
    """

    def __init__(
        self, redis_client, keyer: HostKeyer, config: dict[str, Any], dataset: str = "cicids2017"
    ):
        self.redis = redis_client
        self.keyer = keyer
        self.dataset = dataset
        self.prefix = config["key_prefix"]
        self.short_ms = int(config["short_window_s"]) * 1000
        self.long_ms = int(config["long_window_s"]) * 1000
        self.bucket_s = int(config["bucket_s"])
        self.n_buckets = self.long_ms // (self.bucket_s * 1000)
        self.lateness_ms = int(config["allowed_lateness_s"]) * 1000
        factor = int(config["ttl_factor"])
        self.ttl_short = factor * int(config["short_window_s"])
        self.ttl_long = factor * int(config["long_window_s"])
        fields = config["source_fields"]
        self.port_field = fields["dst_port"]
        self.bytes_field = fields["fwd_bytes"]
        self.syn_field = fields["syn_count"]
        self.watermark_ms: int | None = None
        self.late_events = 0

    def _z(self, kind: str, host_key: str) -> str:
        return f"{self.prefix}z:{kind}:{host_key}"

    def _agg(self, host_key: str, bucket: int) -> str:
        return f"{self.prefix}agg:{host_key}:{bucket}"

    def compute(self, records: list[dict[str, Any]]) -> list[dict[str, float]]:
        """
        Update window state with a batch and return each record's features, in input order.

        Parameters
        ----------
        records : list of dict
            Decoded flows with `event_id`, `event_time` (ms), `src_ip`, `dst_ip`, and the
            configured source fields.

        Returns
        -------
        features : list of dict
            One dict per record with the keys of BEHAVIOR_FEATURES.
        """
        if not records:
            return []
        prepared = []
        late = 0
        for record in records:
            t = int(record["event_time"])
            if self.watermark_ms is not None and t < self.watermark_ms - self.lateness_ms:
                late += 1
            self.watermark_ms = t if self.watermark_ms is None else max(self.watermark_ms, t)
            prepared.append(
                (
                    record["event_id"],
                    t,
                    self.keyer.key(str(record["src_ip"])),
                    self.keyer.key(str(record["dst_ip"])),
                    int(record[self.port_field]),
                    float(record[self.bytes_field]),
                    int(float(record[self.syn_field]) > 0),
                )
            )
        if late:
            # late events still update state; they are only counted
            self.late_events += late
            LATE_EVENTS.labels(dataset=self.dataset).inc(late)

        write = self.redis.pipeline(transaction=False)
        for event_id, t, src, dst, port, fwd_bytes, syn in prepared:
            b = bucket_of(t, self.bucket_s)
            z_src, z_dst = self._z("src", src), self._z("dst", dst)
            write.zadd(z_src, {event_id: t})
            write.zremrangebyscore(z_src, "-inf", t - self.long_ms - self.lateness_ms)
            write.expire(z_src, self.ttl_long)
            write.zadd(z_dst, {event_id: t})
            write.zremrangebyscore(z_dst, "-inf", t - self.short_ms - self.lateness_ms)
            write.expire(z_dst, self.ttl_short)
            for kind, member in (("ports", str(port)), ("hosts", dst)):
                key = hll_key(self.prefix, kind, src, b)
                write.pfadd(key, member)
                write.expire(key, self.ttl_long)
            agg = self._agg(src, b)
            write.hincrbyfloat(agg, "fwd_bytes", fwd_bytes)
            write.hincrby(agg, "n", 1)
            write.hincrby(agg, "syn", syn)
            write.expire(agg, self.ttl_long)
        write.execute()

        newest_bucket: dict[str, int] = {}
        for _, t, src, *_ in prepared:
            newest_bucket[src] = max(newest_bucket.get(src, -1), bucket_of(t, self.bucket_s))
        read = self.redis.pipeline(transaction=False)
        for _, t, src, dst, *_ in prepared:
            read.zcount(self._z("src", src), f"({t - self.short_ms}", t)
            read.zcount(self._z("src", src), f"({t - self.long_ms}", t)
            read.zcount(self._z("dst", dst), f"({t - self.short_ms}", t)
        sources = sorted(newest_bucket)
        for src in sources:
            buckets = bucket_range(newest_bucket[src], self.n_buckets)
            merged_count(read, [hll_key(self.prefix, "ports", src, b) for b in buckets])
            merged_count(read, [hll_key(self.prefix, "hosts", src, b) for b in buckets])
            for b in buckets:
                read.hmget(self._agg(src, b), "fwd_bytes", "n", "syn")
        results = read.execute()

        per_event = 3
        per_source = 2 + self.n_buckets
        offset = per_event * len(prepared)
        source_features = {}
        for i, src in enumerate(sources):
            block = results[offset + i * per_source : offset + (i + 1) * per_source]
            total_bytes = total_n = total_syn = 0.0
            for fwd_bytes, n, syn in block[2:]:
                total_bytes += float(fwd_bytes or 0.0)
                total_n += float(n or 0.0)
                total_syn += float(syn or 0.0)
            source_features[src] = {
                "src_unique_dst_ports_60s": float(block[0]),
                "src_unique_dst_hosts_60s": float(block[1]),
                "src_mean_fwd_bytes_60s": total_bytes / total_n if total_n else 0.0,
                "src_syn_flow_share_60s": total_syn / total_n if total_n else 0.0,
            }
        features = []
        for j, (_, _, src, *_rest) in enumerate(prepared):
            counts = results[per_event * j : per_event * (j + 1)]
            features.append(
                {
                    "src_conn_count_10s": float(counts[0]),
                    "src_conn_count_60s": float(counts[1]),
                    "dst_conn_count_10s": float(counts[2]),
                    **source_features[src],
                }
            )
        return features


def build_stream_feature_fn(dataset: str, redis_client, settings):
    """Return the consumer's per-batch feature function backed by a StreamFeatureComputer."""
    config = load_config(settings.config_path("stream_features.yaml")).data
    computer = StreamFeatureComputer(
        redis_client, HostKeyer(settings.ip_hash_salt.get_secret_value()), config, dataset
    )
    return computer.compute
