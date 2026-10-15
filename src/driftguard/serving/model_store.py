"""Alias-based model loading with warmup and atomic swaps (SPEC 4.2)."""

import json
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from driftguard.common.logging import get_logger
from driftguard.common.metrics import MODEL_INFO
from driftguard.common.mlflow_client import (
    CHALLENGER,
    CHAMPION,
    FEATURES_FILE,
    MODEL_FILE,
    Registry,
    model_name,
)
from driftguard.training.export import onnx_scores, onnx_session

# ============= Constants =============

logger = get_logger("serving.model_store")
SERVED_ALIASES = (CHAMPION, CHALLENGER)


# ============= Models =============


@dataclass
class LoadedModel:
    """An ONNX session with its version, threshold, and input feature order."""

    alias: str
    version: str
    threshold: float
    features: list[str]
    session: object
    _index_cache: dict[tuple[str, ...], np.ndarray] = field(default_factory=dict)

    def column_index(self, request_features: tuple[str, ...]) -> np.ndarray:
        """Indices that select this model's inputs from a request's feature order."""
        cached = self._index_cache.get(request_features)
        if cached is None:
            position = {name: i for i, name in enumerate(request_features)}
            missing = [name for name in self.features if name not in position]
            if missing:
                raise KeyError(f"request lacks model features: {missing[:5]}")
            cached = np.array([position[name] for name in self.features], dtype=np.int64)
            self._index_cache[request_features] = cached
        return cached

    def score(self, request_features: tuple[str, ...], matrix: np.ndarray) -> np.ndarray:
        """Score a request matrix of shape (n, len(request_features)); returns shape (n,)."""
        return onnx_scores(self.session, matrix[:, self.column_index(request_features)])


def load_model(
    alias: str, version: str, threshold: float, model_dir: Path, warmup_rows: int
) -> LoadedModel:
    """Create a session from a downloaded model directory and run a warmup batch."""
    features = json.loads((model_dir / FEATURES_FILE).read_text())
    session = onnx_session(model_dir / MODEL_FILE)
    onnx_scores(session, np.zeros((warmup_rows, len(features)), dtype=np.float32))
    return LoadedModel(alias, version, threshold, features, session)


# ============= Store =============


class ModelStore:
    """
    Holds the models currently served per alias and swaps them atomically.

    A new version is downloaded, loaded, and warmed up in the caller's thread (the watcher's),
    then the dictionary entry is replaced under a lock. Requests read the entry once, so
    in-flight requests finish on the session they started with.

    Parameters
    ----------
    registry : Registry
        Model registry.
    dataset : str
        Dataset served.
    warmup_rows : int
        Rows in the warmup batch.
    """

    def __init__(self, registry: Registry, dataset: str, warmup_rows: int = 256):
        self.registry = registry
        self.dataset = dataset
        self.warmup_rows = warmup_rows
        self._models: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    def get(self, alias: str) -> LoadedModel | None:
        """Return the model served for an alias, or None."""
        with self._lock:
            return self._models.get(alias)

    def ready(self) -> bool:
        """True once the champion is loaded and warm."""
        return self.get(CHAMPION) is not None

    def refresh(self) -> list[str]:
        """
        Synchronize served models with the registry aliases.

        Returns
        -------
        changed : list of str
            Aliases whose model was swapped or removed.
        """
        changed = []
        name = model_name(self.dataset)
        for alias in SERVED_ALIASES:
            ref = self.registry.get_alias(name, alias)
            current = self.get(alias)
            if ref is None:
                if current is not None:
                    with self._lock:
                        self._models.pop(alias, None)
                    MODEL_INFO.labels(
                        dataset=self.dataset, alias=alias, version=current.version
                    ).set(0)
                    changed.append(alias)
                continue
            if current is not None and current.version == ref.version:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                model_dir = self.registry.download(ref, Path(tmp))
                loaded = load_model(alias, ref.version, ref.threshold, model_dir, self.warmup_rows)
            with self._lock:
                self._models[alias] = loaded
            if current is not None:
                MODEL_INFO.labels(dataset=self.dataset, alias=alias, version=current.version).set(0)
            MODEL_INFO.labels(dataset=self.dataset, alias=alias, version=ref.version).set(1)
            logger.info(
                "model_swapped",
                alias=alias,
                version=ref.version,
                previous=current.version if current else None,
            )
            changed.append(alias)
        return changed


class AliasWatcher:
    """Background thread polling the registry every `poll_s` seconds (SPEC: 10 s)."""

    def __init__(self, store: ModelStore, poll_s: float = 10.0):
        self.store = store
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.thread = threading.Thread(target=self._run, name="alias-watcher", daemon=True)

    def start(self) -> None:
        """Start polling."""
        self.thread.start()

    def poke(self) -> None:
        """Refresh now instead of waiting for the next poll (used after admin actions)."""
        self._wake.set()

    def stop(self) -> None:
        """Stop polling."""
        self._stop.set()
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.store.refresh()
            except Exception as exc:  # noqa: BLE001
                # registry or artifact errors must not kill the watcher; the old model keeps serving
                logger.error("alias_refresh_failed", error=str(exc))
            self._wake.wait(self.poll_s)
            self._wake.clear()
            time.sleep(0)
