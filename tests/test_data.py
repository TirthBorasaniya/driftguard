"""Tests for the retained SafeLabelEncoder utility."""

import numpy as np
import pandas as pd

from src.data.encoders import SafeLabelEncoder


class TestSafeLabelEncoder:
    def test_fit_transform_known(self):
        enc = SafeLabelEncoder()
        s = pd.Series(["a", "b", "c", "a"])
        enc.fit(s)
        result = enc.transform(s)
        assert set(result.tolist()) == {0, 1, 2}

    def test_unseen_maps_to_negative_one(self):
        enc = SafeLabelEncoder()
        enc.fit(pd.Series(["a", "b"]))
        result = enc.transform(pd.Series(["a", "z", "unknown_val"]))
        assert result.iloc[1] == -1
        assert result.iloc[2] == -1

    def test_null_maps_to_negative_one(self):
        enc = SafeLabelEncoder()
        enc.fit(pd.Series(["a", "b"]))
        result = enc.transform(pd.Series(["a", None, np.nan]))
        assert result.iloc[1] == -1
        assert result.iloc[2] == -1

    def test_save_load(self, tmp_path):
        enc = SafeLabelEncoder()
        enc.fit(pd.Series(["x", "y", "z"]))
        path = tmp_path / "enc.pkl"
        enc.save(path)
        loaded = SafeLabelEncoder.load(path)
        result = loaded.transform(pd.Series(["x", "new"]))
        assert result.iloc[0] >= 0
        assert result.iloc[1] == -1
