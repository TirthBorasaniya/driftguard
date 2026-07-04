"""Pytest fixtures and markers for the network anomaly detection test suite."""

import numpy as np
import pandas as pd
import pytest

from src.config import ENTITY_COL, REFERENCE_CAPTURE_FILE, TARGET_COL
from src.features.engineering import FEATURE_COLS


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_data: marks tests that need source CSVs (skipped in CI)"
    )


requires_data = pytest.mark.skipif(
    not REFERENCE_CAPTURE_FILE.exists(),
    reason="CICIDS2017 capture files not found. Download from unb.ca/cic/datasets/ids-2017.html.",
)


@pytest.fixture
def sample_flow_event():
    """Single network flow event dict carrying the raw schema fields."""
    return {
        "event_id": "evt-0001",
        "flow_id": "192.168.10.5-52.6.13.28-49158-443-6",
        "timestamp_utc": 1_700_000_000_000,
        "src_ip": "192.168.10.5",
        "dst_ip": "52.6.13.28",
        "src_port": 49158,
        "dst_port": 443,
        "protocol": 6,
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
        "label": "BENIGN",
        "label_binary": 0,
    }


@pytest.fixture
def sample_flow_request():
    """Request payload for the /predict endpoint (NetworkFlowRequest fields)."""
    return {
        "flow_duration": 100000.0,
        "flow_bytes_per_sec": 5000.0,
        "flow_packets_per_sec": 50.0,
        "total_fwd_packets": 10.0,
        "total_bwd_packets": 8.0,
        "packet_length_mean": 120.0,
        "packet_length_std": 30.0,
        "flow_iat_mean": 2000.0,
        "syn_flag_count": 1.0,
        "src_ip": "192.168.10.5",
        "flow_id": "192.168.10.5-52.6.13.28-49158-443-6",
    }


@pytest.fixture
def sample_feature_df():
    """Processed feature frame: FEATURE_COLS plus target, entity, and event_timestamp."""
    np.random.seed(42)
    n = 400
    df = pd.DataFrame({col: np.abs(np.random.rand(n)) * 100 for col in FEATURE_COLS})
    df[TARGET_COL] = np.random.choice([0, 1], n, p=[0.8, 0.2])
    df[ENTITY_COL] = np.random.choice(
        ["192.168.10.5", "192.168.10.8", "172.16.0.1"], n
    )
    df["event_timestamp"] = pd.date_range("2017-07-03 09:00:00", periods=n, freq="1min")
    return df
