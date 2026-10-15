"""Tests for the orchestrator trigger logic, offline store, training window, and retraining run."""

import json
from pathlib import Path

import fakeredis
import numpy as np
import pandas as pd
import pytest

from driftguard.common.config import load_config
from driftguard.common.mlflow_client import CHALLENGER, CHAMPION, ModelRef, model_name
from driftguard.common.promotions import append_promotion, read_promotions
from driftguard.features.offline import OfflineStore
from driftguard.orchestrator.triggers import (
    RESULT_KEY,
    TriggerStore,
    handle_drift_event,
    reconcile_runs,
)
from driftguard.training.flow import (
    STATUS_REGISTERED,
    STATUS_REJECTED_DQ,
    RetrainDeps,
    run_retraining,
)
from driftguard.training.validate import ValidationReport
from driftguard.training.window import (
    build_training_window,
    data_hash,
    direct_historical,
    labeled_events,
    split_window,
)

pytestmark = pytest.mark.unit
EVENT = {
    "dataset": "cicids2017",
    "window_id": "window_00007",
    "detected_at": 5,
    "signals": {"psi_max": 0.4, "window_end_ms": 123.0},
    "severity": "LOW",
    "policy_version": 3,
}


# ----- orchestrator -----


def test_trigger_submits_with_timeout_zero_semantics(tmp_path):
    store, redis = TriggerStore(tmp_path / "o.db"), fakeredis.FakeRedis(decode_responses=True)
    submitted = []
    outcome = handle_drift_event(EVENT, store, redis, lambda p: submitted.append(p) or "run-1")
    assert outcome.status == "submitted" and outcome.flow_run_id == "run-1"
    assert submitted[0]["window_end_ms"] == 123 and submitted[0]["trigger_id"] == outcome.trigger_id
    row = store.all()[0]
    assert set(row) >= {"trigger_id", "dataset", "window_id", "flow_run_id", "status", "created_at"}


def test_trigger_skipped_while_paused(tmp_path):
    store, redis = TriggerStore(tmp_path / "o.db"), fakeredis.FakeRedis(decode_responses=True)
    redis.set("dg:retrain:paused:cicids2017", "data_quality")
    outcome = handle_drift_event(EVENT, store, redis, lambda p: pytest.fail("must not submit"))
    assert outcome.status == "skipped_paused"


def test_submit_failure_recorded(tmp_path):
    store, redis = TriggerStore(tmp_path / "o.db"), fakeredis.FakeRedis(decode_responses=True)

    def boom(_params):
        raise ConnectionError("prefect down")

    assert handle_drift_event(EVENT, store, redis, boom).status == "submit_failed"


def test_reconcile_uses_flow_result(tmp_path):
    store, redis = TriggerStore(tmp_path / "o.db"), fakeredis.FakeRedis(decode_responses=True)
    first = handle_drift_event(EVENT, store, redis, lambda p: "r1")
    second = handle_drift_event(EVENT, store, redis, lambda p: "r2")
    redis.set(RESULT_KEY.format(trigger_id=first.trigger_id), STATUS_REJECTED_DQ)
    states = {"r1": "COMPLETED", "r2": "CANCELLED"}
    result = lambda tid: redis.get(RESULT_KEY.format(trigger_id=tid))  # noqa: E731
    assert reconcile_runs(store, states.get, result) == 2
    statuses = {r["trigger_id"]: r["status"] for r in store.all()}
    assert statuses[first.trigger_id] == STATUS_REJECTED_DQ
    assert statuses[second.trigger_id] == "cancelled"


def test_promotions_log_is_append_only(tmp_path):
    path = tmp_path / "p.jsonl"
    append_promotion(path, {"dataset": "d", "action": "bootstrap", "champion": "1"})
    append_promotion(path, {"dataset": "d", "action": "promote", "champion": "2"})
    rows = read_promotions(path)
    assert [r["action"] for r in rows] == ["bootstrap", "promote"] and "ts" in rows[0]


# ----- offline store and window -----


def _populate(offline, n=400, seed=0, label_delay_ms=10):
    from driftguard.data.registry import feature_names

    names = feature_names("cicids2017")
    rng = np.random.default_rng(seed)
    times = 1_499_083_200_000 + np.arange(n) * 1000
    labels = (rng.random(n) < 0.2).astype(int)
    values = rng.normal(size=(n, len(names))).astype("float32")
    values[labels == 1, 0] += 3.0
    features = pd.DataFrame(values, columns=names)
    features.insert(0, "event_id", [f"e{i:05d}" for i in range(n)])
    features.insert(1, "event_timestamp", pd.to_datetime(times, unit="ms", utc=True))
    offline.append("cicids2017", "flow_features", features.iloc[: n // 2])
    offline.append("cicids2017", "flow_features", features.iloc[n // 2 :])
    label_frame = pd.DataFrame(
        {
            "event_id": features["event_id"],
            "label": labels,
            "label_family": np.where(labels == 1, "DDoS", "BENIGN"),
            "label_time": times + label_delay_ms,
        }
    )
    label_frame["label_timestamp"] = pd.to_datetime(label_frame["label_time"], unit="ms", utc=True)
    offline.append("cicids2017", "labels", label_frame)
    offline.append("cicids2017", "labels", label_frame.iloc[:5])  # at-least-once duplicates
    return names, times


def test_labeled_events_respects_window_end_and_dedupes(tmp_path):
    offline = OfflineStore(tmp_path)
    _, times = _populate(offline)
    events = labeled_events(offline, "cicids2017", int(times[99]) + 10)
    assert len(events) == 100 and events["event_id"].is_unique
    assert events["event_timestamp"].is_monotonic_increasing


def test_window_point_in_time_and_splits(tmp_path):
    offline = OfflineStore(tmp_path)
    names, times = _populate(offline)
    refs = [f"flow_features_cicids2017:{n}" for n in names]
    frame = build_training_window(
        offline,
        "cicids2017",
        int(times[-1]) + 10,
        300,
        refs,
        direct_historical(offline, "cicids2017"),
    )
    assert len(frame) == 300 and frame["event_id"].iloc[0] == "e00100"
    splits = split_window(frame, 0.2, 0.1)
    assert len(splits.holdout) == 60 and len(splits.validation) == 24
    assert splits.fit["event_timestamp"].max() < splits.validation["event_timestamp"].min()
    assert splits.validation["event_timestamp"].max() < splits.holdout["event_timestamp"].min()
    assert data_hash(frame) == data_hash(frame.sample(frac=1.0, random_state=1))


# ----- retraining run -----


class FakeRegistry:
    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.aliases, self.versions, self.tags = {}, {}, {}

    def get_alias(self, name, alias):
        version = self.aliases.get((name, alias))
        if version is None:
            return None
        return ModelRef(name, version, self.tags[version], f"mem://{version}")

    def set_alias(self, name, alias, version):
        self.aliases[(name, alias)] = version

    def delete_alias(self, name, alias):
        self.aliases.pop((name, alias), None)

    def download(self, ref, dst):
        out = dst / ref.version
        out.mkdir(parents=True, exist_ok=True)
        for file_name, payload in self.versions[ref.version].items():
            (out / file_name).write_bytes(payload)
        return out

    def register(self, dataset, artifacts, tags, metrics, params):
        version = str(len(self.versions) + 1)
        self.versions[version] = artifacts
        self.tags[version] = {k: str(v) for k, v in tags.items()}
        return version


def _deps(tmp_path, registry, gx_ok=True):
    offline = OfflineStore(tmp_path / "off")
    names, _ = _populate(offline, n=1500, seed=4)
    refs = [f"flow_features_cicids2017:{n}" for n in names]
    model_cfg = load_config("configs/model.yaml").data
    model_cfg = {
        **model_cfg,
        "lightgbm": {**model_cfg["lightgbm"], "n_estimators": 40, "n_jobs": 1},
    }
    statuses, modes = [], []

    class GX:
        def validate(self, frame):
            return ValidationReport(gx_ok, [] if gx_ok else ["bounds:x"], 1)

    deps = RetrainDeps(
        registry=registry,
        build_window=lambda end, n: build_training_window(
            offline, "cicids2017", end, n, refs, direct_historical(offline, "cicids2017")
        ),
        validator=GX(),
        set_rollout_mode=lambda ds, mode: modes.append(mode),
        report_status=lambda tid, status: statuses.append(status),
        model_config=model_cfg,
        retrain_config={
            **load_config("configs/retrain.yaml").data,
            "train_window_labeled": 1200,
            "min_holdout_positives": 5,
            "reference_max_rows": 500,
        },
        features=names,
    )
    return deps, statuses, modes


PARAMS = {
    "dataset": "cicids2017",
    "trigger_id": "t1",
    "window_id": "w",
    "window_end_ms": 1_499_083_200_000 + 2_000_000,
}


def test_run_retraining_registers_challenger_and_starts_shadow(tmp_path):
    registry = FakeRegistry(tmp_path)
    deps, statuses, modes = _deps(tmp_path, registry)
    summary = run_retraining(PARAMS, deps)
    assert summary["status"] == STATUS_REGISTERED and statuses == [STATUS_REGISTERED]
    assert modes == ["shadow"]
    name = model_name("cicids2017")
    version = registry.aliases[(name, CHALLENGER)]
    tags = registry.tags[version]
    for key in ("trigger_id", "window_end", "train_rows", "holdout_rows", "data_hash", "threshold"):
        assert key in tags
    assert set(registry.versions[version]) == {"model.onnx", "features.json", "reference.parquet"}
    # a second run compares against an existing champion on the same holdout
    registry.set_alias(name, CHAMPION, version)
    registry.delete_alias(name, CHALLENGER)  # the first stage ended
    second = run_retraining({**PARAMS, "trigger_id": "t2"}, deps)
    assert second["holdout"]["champion"] is not None
    assert json.loads(registry.tags[second["challenger_version"]]["holdout_comparison"])


def test_run_retraining_rejects_on_data_quality(tmp_path):
    registry = FakeRegistry(tmp_path)
    deps, statuses, modes = _deps(tmp_path, registry, gx_ok=False)
    assert run_retraining(PARAMS, deps)["status"] == STATUS_REJECTED_DQ
    assert statuses == [STATUS_REJECTED_DQ] and not modes and not registry.versions


def test_gx_validator_bounds_and_mostly():
    from driftguard.training.validate import GXValidator, fit_bounds

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(2000, 3)), columns=["a", "b", "c"])
    validator = GXValidator(fit_bounds(frame, ["a", "b", "c"]), mostly=0.99, min_rows=100)
    assert validator.validate(frame).success
    broken = frame.copy()
    broken.loc[:100, "a"] = 1e6
    report = validator.validate(broken)
    assert not report.success and any(":a" in item for item in report.failed)
    assert not validator.validate(frame.iloc[:50]).success


def test_trigger_skipped_during_active_rollout(tmp_path):
    """Regression (M2): a new run replaced the challenger under an in-flight shadow or canary."""
    store, redis = TriggerStore(tmp_path / "o.db"), fakeredis.FakeRedis(decode_responses=True)
    for mode in ("shadow", "canary"):
        redis.set("dg:rollout:cicids2017", mode)
        outcome = handle_drift_event(
            EVENT, store, redis, lambda p: pytest.fail("must not submit while rolling out")
        )
        assert outcome.status == "skipped_active_rollout"
    redis.set("dg:rollout:cicids2017", "champion_only")
    assert handle_drift_event(EVENT, store, redis, lambda p: "run-9").status == "submitted"


def test_retraining_flow_refuses_to_replace_an_existing_challenger(tmp_path):
    registry = FakeRegistry(tmp_path)
    deps, statuses, modes = _deps(tmp_path, registry)
    first = run_retraining(PARAMS, deps)
    assert first["status"] == STATUS_REGISTERED
    versions_before = dict(registry.versions)
    second = run_retraining({**PARAMS, "trigger_id": "t2"}, deps)
    assert second["status"] == "skipped_active_challenger"
    assert registry.versions == versions_before and modes == ["shadow"]
    assert statuses[-1] == "skipped_active_challenger"
