"""Shared test doubles: an in-memory model registry and small real ONNX models."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.common.mlflow_client import FEATURES_FILE, MODEL_FILE, REFERENCE_FILE, ModelRef
from driftguard.training.export import to_onnx_bytes
from driftguard.training.train import train_model

PARAMS = {
    "objective": "binary",
    "n_estimators": 20,
    "learning_rate": 0.2,
    "num_leaves": 7,
    "min_child_samples": 5,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
    "n_jobs": 1,
}


class FakeRegistry:
    """In-memory registry with the same interface as MlflowRegistry."""

    def __init__(self):
        self.aliases: dict[tuple[str, str], str] = {}
        self.versions: dict[str, dict[str, bytes]] = {}
        self.tags: dict[str, dict[str, str]] = {}

    def get_alias(self, name, alias):
        version = self.aliases.get((name, alias))
        if version is None:
            return None
        return ModelRef(name, version, self.tags[version], f"mem://{version}")

    def set_alias(self, name, alias, version):
        self.aliases[(name, alias)] = str(version)

    def delete_alias(self, name, alias):
        self.aliases.pop((name, alias), None)

    def download(self, ref, dst: Path) -> Path:
        out = Path(dst) / f"v{ref.version}"
        out.mkdir(parents=True, exist_ok=True)
        for file_name, payload in self.versions[ref.version].items():
            (out / file_name).write_bytes(payload)
        return out

    def register(self, dataset, artifacts, tags, metrics=None, params=None):
        version = str(len(self.versions) + 1)
        self.versions[version] = artifacts
        self.tags[version] = {k: str(v) for k, v in tags.items()}
        return version


def model_artifacts(
    features: list[str], seed: int = 0, flip: bool = False, n: int = 600
) -> dict[str, bytes]:
    """Train a tiny model on synthetic data and return its registry artifacts."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(features))).astype(np.float32)
    y = (x[:, 0] > 0.5).astype(int)
    if flip:
        y = 1 - y
    model, _ = train_model(x, y, PARAMS, seed)
    reference = pd.DataFrame(x[:200], columns=features)
    reference["score"] = model.predict_proba(x[:200])[:, 1]
    buffer = io.BytesIO()
    reference.to_parquet(buffer, index=False)
    return {
        MODEL_FILE: to_onnx_bytes(model, len(features), split=10),
        FEATURES_FILE: json.dumps(features).encode(),
        REFERENCE_FILE: buffer.getvalue(),
    }
