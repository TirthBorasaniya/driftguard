"""Tests for the CICIDS2017 adapter: allowlist, labels, attempted policy, cleaning, ids."""

import uuid

import numpy as np
import pandas as pd
import pytest

from driftguard.data import cicids2017
from driftguard.data.cicids2017 import drop_nonfinite, map_labels, model_matrix, order_stream
from driftguard.data.ids import make_event_id
from driftguard.data.registry import dataset_config, feature_names, feature_specs

pytestmark = pytest.mark.unit

IDENTIFIER_RAW_COLUMNS = {
    "id",
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Timestamp",
    "Label",
    "Attempted Category",
    "Source IP",
    "Source Port",
    "Destination IP",
}


def test_allowlist_excludes_identifiers_for_both_sources():
    for source in ("improved", "original"):
        raw_columns = {spec[source] for spec in feature_specs("cicids2017", source)}
        assert not raw_columns & IDENTIFIER_RAW_COLUMNS
        names = set(feature_names("cicids2017", source))
        assert not names & set(dataset_config("cicids2017")["excluded_columns"])
        assert not names & {"src_ip", "dst_ip", "src_port", "timestamp", "flow_id", "label"}
    # destination port is kept on purpose; E1 ablates it
    assert "dst_port" in feature_names("cicids2017")


def test_model_matrix_rejects_excluded_columns():
    frame = pd.DataFrame({"flow_duration": [1.0], "src_ip": ["1.2.3.4"]})
    with pytest.raises(ValueError):
        model_matrix(frame, ["flow_duration", "src_ip"])
    assert model_matrix(frame, ["flow_duration"]).dtype == np.float32


@pytest.mark.parametrize(
    ("policy", "expected_labels", "expected_keep"),
    [
        ("benign", [0, 1, 0, 1], [True] * 4),
        ("attack", [0, 1, 1, 1], [True] * 4),
        ("drop", [0, 1, 1, 1], [True, True, False, True]),
    ],
)
def test_label_mapping_and_attempted_policy(policy, expected_labels, expected_keep):
    raw = pd.Series(["BENIGN", "DoS Hulk", "DoS Hulk - Attempted", "Portscan"])
    mapped = map_labels(raw, policy)
    assert mapped["label"].tolist() == expected_labels
    assert mapped["keep"].tolist() == expected_keep
    assert mapped["label_family"].tolist() == ["BENIGN", "DoS Hulk", "DoS Hulk", "Portscan"]
    assert mapped["attempted"].tolist() == [False, False, True, False]


def test_original_release_dash_normalized():
    mapped = map_labels(pd.Series(["Web Attack � Brute Force", "Web Attack \x96 XSS"]), "benign")
    assert mapped["label_family"].tolist() == ["Web Attack - Brute Force", "Web Attack - XSS"]
    assert mapped["label"].tolist() == [1, 1]


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        map_labels(pd.Series(["BENIGN"]), "maybe")


def test_drop_nonfinite_counts_rows():
    frame = pd.DataFrame({"a": [1.0, np.inf, 3.0, np.nan], "b": [1.0, 2.0, -np.inf, 4.0]})
    cleaned, dropped = drop_nonfinite(frame, ["a", "b"])
    assert dropped == 3
    assert cleaned["a"].tolist() == [1.0]


def test_event_id_is_deterministic_uuid5():
    first = make_event_id("cicids2017", "monday.csv", 7)
    assert first == make_event_id("cicids2017", "monday.csv", 7)
    assert first != make_event_id("cicids2017", "monday.csv", 8)
    assert first == str(uuid.uuid5(uuid.NAMESPACE_URL, "cicids2017:monday.csv:7"))


def test_order_stream_sorts_by_day_then_time():
    frame = pd.DataFrame(
        {"day": ["tuesday", "monday", "monday"], "event_time": [1, 9, 5], "row_index": [0, 1, 2]}
    )
    ordered = order_stream(frame, ["monday", "tuesday"])
    assert ordered["event_time"].tolist() == [5, 9, 1]


def test_load_day_from_synthetic_csv(tmp_path, monkeypatch):
    specs = feature_specs("cicids2017", "improved")
    rows = 6
    data = {spec["improved"]: np.arange(rows, dtype=float) for spec in specs}
    data["Flow Bytes/s"] = [1.0, np.inf, 2.0, 3.0, 4.0, 5.0]
    data.update(
        {
            "id": range(1, rows + 1),
            "Flow ID": ["f"] * rows,
            "Src IP": ["10.0.0.1"] * rows,
            "Src Port": [1] * rows,
            "Dst IP": ["10.0.0.2"] * rows,
            "Timestamp": [f"2017-07-03 12:00:0{i}.000001" for i in range(rows)],
            "Label": ["BENIGN", "BENIGN", "DoS Hulk", "DoS Hulk - Attempted", "BENIGN", "DDoS"],
            "Attempted Category": [-1, -1, -1, 0, -1, -1],
        }
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(data).to_csv(raw_dir / "monday.csv", index=False)
    config = dataset_config("cicids2017")
    patched = dict(config.data)
    patched["sources"] = {
        **config["sources"],
        "improved": {**config["sources"]["improved"], "raw_dir": str(raw_dir)},
    }
    monkeypatch.setattr(cicids2017, "REPO_ROOT", tmp_path)
    loaded = type(config)(path=config.path, sha256=config.sha256, data=patched)

    frame, report = cicids2017.load_day("monday", loaded, attempted_policy="drop")
    assert report == {
        "rows_read": 6,
        "dropped_attempted": 1,
        "dropped_nonfinite": 1,
        "rows_kept": 4,
    }
    assert frame["label"].tolist() == [0, 1, 0, 1]
    assert frame["row_index"].tolist() == [0, 2, 4, 5]
    assert frame["event_id"].iloc[0] == make_event_id("cicids2017", "monday.csv", 0)
    assert frame[feature_names("cicids2017")].dtypes.unique().tolist() == [np.float32]
    assert frame["event_time"].iloc[0] == 1499083200000
