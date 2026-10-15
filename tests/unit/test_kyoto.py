"""Tests for the Kyoto adapter: leakage exclusions, labels, bins, categories, month windows."""

import numpy as np
import pandas as pd
import pytest

from driftguard.data import kyoto
from driftguard.data.registry import dataset_config, feature_names, stream_features_enabled
from driftguard.monitor.windows import month_id

pytestmark = pytest.mark.unit


def raw_frame(n=6):
    rows = {
        "0": ["c00", "c015", "c0100", "c01", "c02", "c03"][:n],
        "1": ["http", "dns", "brand-new-service", "ssh", "other", "smtp"][:n],
        "2": ["c20"] * n,
        "3": ["c30", "c357", "c31", "c32", "c33", "c34"][:n],
        "4": ["4"] * n,
        "5": ["1.0"] * n,
        "6": ["0.0"] * n,
        "7": ["0.0"] * n,
        "8": ["90"] * n,
        "9": ["100"] * n,
        "10": ["0.0"] * n,
        "11": ["0.0"] * n,
        "12": ["1.0"] * n,
        "13": ["SF", "S0", "REJ", "SF", "SF", "XX"][:n],
        "14": [
            "20061226 22:44:18",
            "20061229 03:40:01",
            "20061203 21:33:14",
            "20061101 00:00:14",
            "not-a-time",
            "20061231 23:57:23",
        ][:n],
        "15": ["21192(1)"] * n,
        "16": ["Win.Worm(1)"] * n,
        "17": ["130(1)"] * n,
        "18": ["1", "-1", "-2", "1", "1", "-1"][:n],
        "19": ["tcp", "udp", "icmp", "tcp", "tcp", "tcp"][:n],
    }
    return pd.DataFrame(rows)


def test_feature_list_excludes_detectors_ips_ports():
    config = dataset_config("kyoto")
    columns = {spec["column"] for spec in config["features"]}
    assert not columns & {"15", "16", "17", "14", "18"}
    names = set(feature_names("kyoto"))
    assert not names & {
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "IDS_detection",
        "Malware_detection",
        "Ashula_detection",
        "label",
    }
    with pytest.raises(ValueError):
        kyoto.check_no_leakage(["4", "15"], config)
    assert not stream_features_enabled("kyoto")


def test_transform_labels_bins_categories_and_ids():
    frame, report = kyoto.transform(raw_frame(), "2006_subset.parquet", dataset_config("kyoto"))
    assert report["rows_read"] == 6 and report["dropped_invalid"] == 1 and len(frame) == 5
    assert frame["label"].tolist() == [0, 1, 1, 0, 1]
    assert frame["label_family"].tolist() == ["normal", "known", "unknown", "normal", "known"]
    assert frame["duration_bin"].tolist() == [0, 15, 100, 1, 3]
    assert frame["service"].iloc[2] == -1 and report["unseen_categories"]["service"] == 1
    assert frame["flag"].iloc[-1] == -1
    assert frame["month"].iloc[0] == "2006-12"
    assert (
        frame["event_id"].iloc[0]
        == kyoto.make_event_ids("kyoto", "2006_subset.parquet", np.array([0]))[0]
    )
    assert frame[feature_names("kyoto")].dtypes.unique().tolist() == [np.float32]
    assert not {"15", "16", "17"} & set(frame.columns)


def test_bad_bins_and_labels_rejected():
    with pytest.raises(ValueError):
        kyoto.parse_bins(pd.Series(["c20"]), "0")
    with pytest.raises(ValueError):
        kyoto.map_labels(pd.Series(["0"]), dataset_config("kyoto")["labels"])


def test_month_id_boundaries():
    assert month_id(1_199_145_599_000) == "2007-12"  # 2007-12-31 23:59:59 UTC
    assert month_id(1_199_145_600_000) == "2008-01"
