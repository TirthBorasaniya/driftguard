"""Tests for stream features: windows, lateness, HLL merges, determinism, IP privacy, Feast push."""

import json
import logging
import tempfile
from pathlib import Path

import fakeredis
import numpy as np
import pandas as pd
import pytest

from driftguard.common.config import load_config
from driftguard.common.logging import configure_logging
from driftguard.data.synthetic import synthetic_flows
from driftguard.features.offline import OfflineStore
from driftguard.streamfeatures.keys import HostKeyer
from driftguard.streamfeatures.offline import replay_features
from driftguard.streamfeatures.windows import BEHAVIOR_FEATURES, StreamFeatureComputer

pytestmark = pytest.mark.unit
SALT = "unit-test-salt-0123456789"
CONFIG = load_config("configs/stream_features.yaml").data


def computer(redis=None):
    redis = redis or fakeredis.FakeRedis(decode_responses=True)
    return StreamFeatureComputer(redis, HostKeyer(SALT), CONFIG), redis


def flow(i, t_s, src="10.0.0.1", dst="10.0.0.2", port=80, fwd=100.0, syn=0):
    return {
        "event_id": f"e{i}",
        "event_time": int(t_s * 1000),
        "src_ip": src,
        "dst_ip": dst,
        "dst_port": float(port),
        "total_length_of_fwd_packet": fwd,
        "syn_flag_count": syn,
    }


def test_conn_counts_use_half_open_windows_including_current():
    comp, _ = computer()
    out = comp.compute([flow(0, 0), flow(1, 5), flow(2, 9), flow(3, 12)])
    assert [r["src_conn_count_10s"] for r in out] == [1, 2, 3, 3]
    assert [r["src_conn_count_60s"] for r in out] == [1, 2, 3, 4]
    later = comp.compute([flow(4, 70)])[0]
    assert later["src_conn_count_60s"] == 2  # (10 s, 70 s] holds t=12 and t=70
    assert later["dst_conn_count_10s"] == 1


def test_counts_exclude_later_events_processed_first():
    comp, _ = computer()
    comp.compute([flow(0, 20)])
    out = comp.compute([flow(1, 15)])[0]
    assert out["src_conn_count_10s"] == 1


def test_late_events_counted_but_still_update_state():
    comp, _ = computer()
    comp.compute([flow(0, 100)])
    out = comp.compute([flow(1, 90)])[0]  # 10 s behind the watermark, lateness 5 s
    assert comp.late_events == 1
    assert out["src_conn_count_10s"] == 1
    assert comp.compute([flow(2, 99)])[0]["src_conn_count_10s"] == 2  # (89, 99] holds t=90, 99


def test_hll_unique_counts_merge_last_six_buckets():
    comp, _ = computer()
    batch = [flow(i, i * 10, port=1000 + i, dst=f"10.1.0.{i}") for i in range(8)]  # 0..70 s
    out = comp.compute(batch)
    # newest bucket is 7 (70 s); buckets 2..7 hold ports 1002..1007
    assert out[-1]["src_unique_dst_ports_60s"] == 6
    assert out[-1]["src_unique_dst_hosts_60s"] == 6
    assert out[0]["src_unique_dst_ports_60s"] == out[-1]["src_unique_dst_ports_60s"]  # per batch


def test_mean_bytes_and_syn_share():
    comp, _ = computer()
    out = comp.compute(
        [
            flow(0, 1, fwd=100, syn=1),
            flow(1, 2, fwd=300, syn=0),
            flow(2, 3, src="10.9.9.9", fwd=50, syn=2),
        ]
    )
    assert out[0]["src_mean_fwd_bytes_60s"] == pytest.approx(200.0)
    assert out[0]["src_syn_flow_share_60s"] == pytest.approx(0.5)
    assert out[2]["src_mean_fwd_bytes_60s"] == pytest.approx(50.0)
    assert list(out[0]) == list(BEHAVIOR_FEATURES)


def test_ttls_bound_every_key():
    comp, redis = computer()
    comp.compute([flow(0, 1)])
    for key in redis.keys("*"):
        assert 0 < redis.ttl(key) <= 120


def test_replay_is_deterministic_and_restores_order():
    frame = synthetic_flows("cicids2017", 2000, seed=3)
    frame["src_ip"] = [f"10.0.0.{i % 7}" for i in range(len(frame))]
    shuffled = frame.sample(frac=1.0, random_state=1)
    first = replay_features(shuffled, computer()[0], 500)
    second = replay_features(shuffled, computer()[0], 500)
    assert first[list(BEHAVIOR_FEATURES)].equals(second[list(BEHAVIOR_FEATURES)])
    assert first.index.equals(shuffled.index)
    in_order = replay_features(frame, computer()[0], 500)
    pd.testing.assert_frame_equal(
        first.loc[frame.index, list(BEHAVIOR_FEATURES)], in_order[list(BEHAVIOR_FEATURES)]
    )


def test_ips_never_stored_logged_or_written(capsys):
    configure_logging("privacy-test")
    comp, redis = computer()
    records = [flow(i, i, src="192.168.77.13", dst="172.20.5.9") for i in range(5)]
    features = comp.compute(records)
    dump = json.dumps({key: str(redis.dump(key)) for key in redis.keys("*")}) + " ".join(
        redis.keys("*")
    )
    assert "192.168.77.13" not in dump and "172.20.5.9" not in dump
    behavior = pd.DataFrame(features)
    assert not {"src_ip", "dst_ip"} & set(behavior.columns)
    from driftguard.consumer.scoring import ScoringClient

    payload = json.dumps(ScoringClient.payload("cicids2017", [("e0", [1.0])], ["f"]))
    assert "192.168" not in payload
    logging.getLogger().handlers[0].flush()
    assert "192.168.77.13" not in capsys.readouterr().out


def test_feast_push_paths():
    """Records the measured behavior that motivates the partitioned-parquet offline write."""
    from feast.data_source import PushMode

    from driftguard.features.store import EVENT_ENTITY, build_store
    from driftguard.streamfeatures.push import BehaviorSink, behavior_objects, push_source_name

    root = Path(tempfile.mkdtemp())
    offline = OfflineStore(root / "offline")
    store = build_store(
        root / "repo", root / "reg.db", {"type": "sqlite", "path": str(root / "online.db")}
    )
    objects = behavior_objects("cicids2017", offline)
    store.apply([EVENT_ENTITY, *objects])
    frame = pd.DataFrame(
        np.ones((3, len(BEHAVIOR_FEATURES)), dtype="float32"), columns=list(BEHAVIOR_FEATURES)
    )
    frame.insert(0, "event_id", ["a", "b", "c"])
    frame.insert(1, "event_timestamp", pd.to_datetime([1, 2, 3], unit="s", utc=True))
    sink = BehaviorSink(
        "cicids2017", offline, lambda name, df: store.push(name, df, to=PushMode.ONLINE)
    )
    sink.write(frame)
    online = store.get_online_features(
        features=["flow_behavior_cicids2017:src_conn_count_10s"], entity_rows=[{"event_id": "b"}]
    ).to_dict()
    assert online["src_conn_count_10s"] == [1.0]
    historical = store.get_historical_features(
        entity_df=frame[["event_id", "event_timestamp"]],
        features=[f"flow_behavior_cicids2017:{n}" for n in BEHAVIOR_FEATURES],
    ).to_df()
    assert len(historical) == 3 and historical["src_syn_flow_share_60s"].tolist() == [1.0] * 3
    # the file offline store cannot append to a partitioned directory source
    with pytest.raises(Exception):  # noqa: B017
        store.push(push_source_name("cicids2017"), frame, to=PushMode.ONLINE_AND_OFFLINE)


def test_empty_salt_rejected():
    with pytest.raises(ValueError):
        HostKeyer("")


def test_consumer_sends_behavior_features_and_writes_behavior_rows():
    import fakeredis as fr

    from driftguard.consumer.dlq import DeadLetterQueue
    from driftguard.consumer.idempotency import IdempotencyStore
    from driftguard.consumer.main import BatchProcessor, Message
    from driftguard.consumer.scoring import ScoreResponse
    from driftguard.consumer.sinks import FeatureSink
    from driftguard.consumer.validation import FlowValidator
    from driftguard.data.registry import feature_names
    from driftguard.producer.replay import flow_record

    seen_names, behavior_frames, base_frames = [], [], []

    class Scorer:
        last_mode, shadow_timeout_s = "champion_only", 1.0

        def score(self, chunk, names):
            seen_names.append(names)
            assert all(len(vec) == len(names) for _, vec in chunk)
            return ScoreResponse(
                "champion_only",
                [
                    {
                        "event_id": eid,
                        "score": 0.1,
                        "decision": 0,
                        "threshold": 0.5,
                        "model_alias": "champion",
                        "model_version": "1",
                    }
                    for eid, _ in chunk
                ],
                1.0,
            )

    class Sink:
        def write(self, frame):
            behavior_frames.append(frame)

    redis = fr.FakeRedis(decode_responses=True)
    comp = StreamFeatureComputer(redis, HostKeyer(SALT), CONFIG)
    proc = BatchProcessor(
        "cicids2017",
        lambda payload: json.loads(payload),
        FlowValidator("cicids2017"),
        IdempotencyStore(redis, "t"),
        FeatureSink("cicids2017", None, lambda name, frame: base_frames.append(frame)),
        Scorer(),
        DeadLetterQueue(lambda topic, **kw: None, "cicids2017"),
        lambda p: None,
        lambda: None,
        stream_features=comp.compute,
        behavior_sink=Sink(),
    )
    rows = synthetic_flows("cicids2017", 20, seed=1).to_dict("records")
    messages = [
        Message(
            None,
            json.dumps(flow_record(r, "cicids2017", feature_names("cicids2017"))).encode(),
            "flows.cicids2017.v1",
            0,
            i,
        )
        for i, r in enumerate(rows)
    ]
    assert proc.process(messages).scored == 20
    assert seen_names[0] == feature_names("cicids2017") + list(BEHAVIOR_FEATURES)
    assert list(behavior_frames[0].columns) == ["event_id", "event_timestamp", *BEHAVIOR_FEATURES]
    assert not {"src_ip", "dst_ip"} & set(base_frames[0].columns)
