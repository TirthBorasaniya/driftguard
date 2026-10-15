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

    Features are computed per event, in event-time order, in a single pipeline per batch (write
    the event, read its features), so they include the current event, exclude later events, and
    do not depend on batch boundaries.

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

        Events are processed in event-time order. For each event the computer writes its state
        and then reads its features inside one pipeline, so a read sees exactly the events at or
        before that event in processing order and nothing later. Pruning happens once per key
        after all reads. The result therefore does not depend on how the stream was split into
        batches, and no event sees the future of its own batch.

        Windows are `ZCOUNT (t - w, t]` for connection counts and the (up to) six 10 s buckets
        ending at the event's own bucket for unique counts, mean forward bytes, and SYN share.

        Parameters
        ----------
        records : list of dict
            Decoded flows with `event_id`, `event_time` (ms), `src_ip`, `dst_ip`, and the
            configured source fields.

        Returns
        -------
        features : list of dict
            One dict per record, in input order, with the keys of BEHAVIOR_FEATURES.
        """
        if not records:
            return []
        prepared = []
        for position, record in enumerate(records):
            prepared.append(
                (
                    int(record["event_time"]),
                    str(record["event_id"]),
                    position,
                    self.keyer.key(str(record["src_ip"])),
                    self.keyer.key(str(record["dst_ip"])),
                    int(record[self.port_field]),
                    float(record[self.bytes_field]),
                    int(float(record[self.syn_field]) > 0),
                )
            )
        prepared.sort(key=lambda item: (item[0], item[1]))
        self._count_late([item[0] for item in prepared])

        pipe = self.redis.pipeline(transaction=False)
        n_commands = 0
        read_start: dict[int, int] = {}
        touched_z_src: set[str] = set()
        touched_z_dst: set[str] = set()
        touched_other: set[str] = set()
        for t, event_id, position, src, dst, port, fwd_bytes, syn in prepared:
            b = bucket_of(t, self.bucket_s)
            z_src, z_dst = self._z("src", src), self._z("dst", dst)
            ports_key = hll_key(self.prefix, "ports", src, b)
            hosts_key = hll_key(self.prefix, "hosts", src, b)
            agg = self._agg(src, b)
            pipe.zadd(z_src, {event_id: t})
            pipe.zadd(z_dst, {event_id: t})
            pipe.pfadd(ports_key, str(port))
            pipe.pfadd(hosts_key, dst)
            pipe.hincrbyfloat(agg, "fwd_bytes", fwd_bytes)
            pipe.hincrby(agg, "n", 1)
            pipe.hincrby(agg, "syn", syn)
            n_commands += 7
            touched_z_src.add(z_src)
            touched_z_dst.add(z_dst)
            touched_other.update((ports_key, hosts_key, agg))

            read_start[position] = n_commands
            buckets = bucket_range(b, self.n_buckets)
            pipe.zcount(z_src, f"({t - self.short_ms}", t)
            pipe.zcount(z_src, f"({t - self.long_ms}", t)
            pipe.zcount(z_dst, f"({t - self.short_ms}", t)
            merged_count(pipe, [hll_key(self.prefix, "ports", src, k) for k in buckets])
            merged_count(pipe, [hll_key(self.prefix, "hosts", src, k) for k in buckets])
            for k in buckets:
                pipe.hmget(self._agg(src, k), "fwd_bytes", "n", "syn")
            n_commands += 5 + len(buckets)

        newest = prepared[-1][0]
        for key in touched_z_src:
            pipe.zremrangebyscore(key, "-inf", newest - self.long_ms - self.lateness_ms)
            pipe.expire(key, self.ttl_long)
        for key in touched_z_dst:
            pipe.zremrangebyscore(key, "-inf", newest - self.short_ms - self.lateness_ms)
            pipe.expire(key, self.ttl_short)
        for key in touched_other:
            pipe.expire(key, self.ttl_long)
        results = pipe.execute()

        features: list[dict[str, float]] = [{} for _ in records]
        for position, start in read_start.items():
            block = results[start : start + 5 + self.n_buckets]
            total_bytes = total_n = total_syn = 0.0
            for fwd_bytes, n, syn in block[5:]:
                total_bytes += float(fwd_bytes or 0.0)
                total_n += float(n or 0.0)
                total_syn += float(syn or 0.0)
            features[position] = {
                "src_conn_count_10s": float(block[0]),
                "src_conn_count_60s": float(block[1]),
                "dst_conn_count_10s": float(block[2]),
                "src_unique_dst_ports_60s": float(block[3]),
                "src_unique_dst_hosts_60s": float(block[4]),
                "src_mean_fwd_bytes_60s": total_bytes / total_n if total_n else 0.0,
                "src_syn_flow_share_60s": total_syn / total_n if total_n else 0.0,
            }
        return features

    def _count_late(self, times: list[int]) -> None:
        """Count events more than the allowed lateness behind the watermark; they still count."""
        late = 0
        for t in times:
            if self.watermark_ms is not None and t < self.watermark_ms - self.lateness_ms:
                late += 1
            self.watermark_ms = t if self.watermark_ms is None else max(self.watermark_ms, t)
        if late:
            self.late_events += late
            LATE_EVENTS.labels(dataset=self.dataset).inc(late)


def build_stream_feature_fn(dataset: str, redis_client, settings):
    """Return the consumer's per-batch feature function backed by a StreamFeatureComputer."""
    config = load_config(settings.config_path("stream_features.yaml")).data
    computer = StreamFeatureComputer(
        redis_client, HostKeyer(settings.ip_hash_salt.get_secret_value()), config, dataset
    )
    return computer.compute
