"""Tests for the CICIDS2017 Kafka producer replay helpers (no broker required)."""

import time

import pandas as pd
import pytest

from src.producer.flow_producer import (
    NETWORK_FLOW_FIELDS,
    TIMESTAMP_FORMAT,
    build_flow_event,
    reindex_timestamps,
)

# ============= reindex_timestamps =============


def test_reindex_first_record_near_now():
    df = pd.DataFrame(
        {"timestamp_raw": ["03/07/2017 09:00:00 AM", "03/07/2017 09:00:05 AM"]}
    )
    out = reindex_timestamps(df, "timestamp_raw", TIMESTAMP_FORMAT)
    now_ms = int(time.time() * 1000)
    first = int(out["timestamp_utc"].iloc[0])
    assert abs(first - now_ms) < 5000


def test_reindex_preserves_relative_spacing():
    df = pd.DataFrame(
        {"timestamp_raw": ["03/07/2017 09:00:00 AM", "03/07/2017 09:00:05 AM"]}
    )
    out = reindex_timestamps(df, "timestamp_raw", TIMESTAMP_FORMAT)
    spacing = int(out["timestamp_utc"].iloc[1]) - int(out["timestamp_utc"].iloc[0])
    assert spacing == pytest.approx(5000, abs=2)


# ============= build_flow_event =============


def test_build_flow_event_has_event_id_and_all_fields():
    row = pd.Series(
        {
            "flow_id": "fid-1",
            "src_ip": "192.168.10.5",
            "dst_ip": "52.6.13.28",
            "label": "BENIGN",
            "timestamp_utc": 1_700_000_000_000,
            "src_port": 49158,
            "dst_port": 443,
            "protocol": 6,
            "label_binary": 0,
            "flow_duration": 100000.0,
            "flow_bytes_per_sec": 5000.0,
            "flow_packets_per_sec": 50.0,
            "total_fwd_packets": 10.0,
            "total_bwd_packets": 8.0,
            "total_length_fwd_packets": 1200.0,
            "total_length_bwd_packets": 960.0,
            "packet_length_mean": 120.0,
            "packet_length_std": 30.0,
            "flow_iat_mean": 2000.0,
            "syn_flag_count": 1.0,
        }
    )
    event = build_flow_event(row)
    assert "event_id" in event and event["event_id"]
    assert set(event.keys()) == set(NETWORK_FLOW_FIELDS)


def test_build_flow_event_types():
    row = pd.Series(
        {f: (1 if f in ("src_port", "dst_port", "protocol", "label_binary", "timestamp_utc")
             else ("BENIGN" if f == "label" else ("ip" if "ip" in f else ("f" if f == "flow_id" else 2.5))))
         for f in NETWORK_FLOW_FIELDS if f != "event_id"}
    )
    event = build_flow_event(row)
    assert isinstance(event["src_port"], int)
    assert isinstance(event["flow_duration"], float)
    assert isinstance(event["src_ip"], str)
