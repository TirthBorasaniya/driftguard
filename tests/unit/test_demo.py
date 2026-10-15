"""Tests for the demo stream builder."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "demo.py"


def load_demo():
    spec = importlib.util.spec_from_file_location("demo_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stream_is_tiled_with_fresh_ids_and_advancing_time_and_shifted_after_onset():
    demo = load_demo()
    stream, onset = demo.build_stream(
        "synthetic", window=500, onset_window=4, total_events=25_000, slice_events=10_000
    )
    assert len(stream) == 25_000 and onset == 2_000
    assert stream["event_id"].is_unique
    assert stream["event_time"].is_monotonic_increasing
    base = demo.load_stream_frame(demo.DATASET, "synthetic")
    first = stream.iloc[:onset]
    assert np.allclose(first["flow_duration"], base["flow_duration"].iloc[:onset])
    ratio = (
        stream["flow_duration"].iloc[onset : onset + 1000].to_numpy()
        / base["flow_duration"].iloc[onset : onset + 1000].to_numpy()
    )
    assert 1.5 < np.median(ratio) < 2.5
    assert (
        stream["label"]
        .iloc[10_000:20_000]
        .equals(base["label"].reset_index(drop=True).rename(None).set_axis(range(10_000, 20_000)))
    )
