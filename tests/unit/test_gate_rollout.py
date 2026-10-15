"""Tests for the bootstrap gate and the rollout controller state machine."""

import fakeredis
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from driftguard.common.config import load_config
from driftguard.common.mlflow_client import CHALLENGER, CHAMPION, PREVIOUS, model_name
from driftguard.common.promotions import read_promotions
from driftguard.orchestrator.rollout import (
    RolloutController,
    evaluate_guardrails,
    guardrail_queries,
)
from driftguard.training.gate import average_precision_rows, paired_bootstrap_delta, shadow_gate

from .fakes import FakeRegistry

pytestmark = pytest.mark.unit
NAME = model_name("cicids2017")


def test_vectorized_ap_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = (rng.random(300) < 0.2).astype(int)
        scores = np.round(rng.random(300), 1)  # many ties
        expected = average_precision_score(y, scores)
        assert average_precision_rows(y[None], scores[None])[0] == pytest.approx(expected)
    assert np.isnan(average_precision_rows(np.zeros((1, 5)), np.ones((1, 5)))[0])


def _scores(n=3000, seed=1, quality=2.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    return y, rng.normal(size=n) + quality * y


def test_gate_decisions():
    y, champion = _scores(quality=1.0)
    rng = np.random.default_rng(9)
    better = rng.normal(size=len(y)) + 3.0 * y
    assert shadow_gate(y, champion, better).decision == "promote"
    assert shadow_gate(y, champion, champion.copy()).decision == "decline"
    worse = rng.normal(size=len(y))
    decision = shadow_gate(y, champion, worse)
    assert decision.decision == "decline" and decision.delta_pr_auc < 0
    deltas = paired_bootstrap_delta(y, champion, champion, 100, seed=1)
    assert np.allclose(deltas, 0.0)


def test_guardrail_evaluation_and_queries():
    rails = {"p99_latency_ms": 50, "error_rate": 0.005, "positive_rate_ratio": [0.5, 1.5]}
    ok = {"p99_latency_s": 0.01, "error_rate": 0.0, "positive_rate_ratio": 1.0}
    assert evaluate_guardrails(ok, rails) == ([], [])
    bad = {"p99_latency_s": 0.06, "error_rate": 0.01, "positive_rate_ratio": 2.0}
    assert evaluate_guardrails(bad, rails) == (
        ["p99_latency", "error_rate", "positive_rate_ratio"],
        [],
    )
    queries = guardrail_queries("cicids2017", "2m")
    assert 'alias="challenger"' in queries["positive_rate_ratio"]
    assert "histogram_quantile(0.99" in queries["p99_latency_s"]
    assert "or vector(0)" in queries["error_rate"]


def test_missing_guardrail_data_is_missing_not_a_pass():
    """Regression (M1): None and NaN used to count as 'not breached'."""
    rails = {"p99_latency_ms": 50, "error_rate": 0.005, "positive_rate_ratio": [0.5, 1.5]}
    values = {"p99_latency_s": None, "error_rate": float("nan"), "positive_rate_ratio": 1.0}
    assert evaluate_guardrails(values, rails) == ([], ["p99_latency", "error_rate"])
    assert evaluate_guardrails({}, rails) == (
        [],
        ["p99_latency", "error_rate", "positive_rate_ratio"],
    )


def _controller(tmp_path, values=None, min_labeled=500, canary_min=100):
    registry = FakeRegistry()
    registry.register("cicids2017", {}, {"threshold": 0.0})
    registry.register("cicids2017", {}, {"threshold": 0.0})
    registry.set_alias(NAME, CHAMPION, "1")
    registry.set_alias(NAME, CHALLENGER, "2")
    redis = fakeredis.FakeRedis(decode_responses=True)
    redis.set("dg:rollout:cicids2017", "shadow")
    config = {
        **load_config("configs/rollout.yaml").data,
        "shadow_min_labeled": min_labeled,
        "canary_min_events": canary_min,
        "bootstrap_resamples": 200,
    }
    values = values or {}
    clock = iter(range(0, 10_000, 31)).__next__
    controller = RolloutController(
        "cicids2017",
        config,
        registry,
        redis,
        tmp_path / "promotions.jsonl",
        query=lambda q: next((v for k, v in values.items() if k in q), None),
        clock=lambda: float(clock()),
    )
    return controller, registry, redis


def _feed(controller, y, champion, challenger, version="2"):
    preds = [
        {
            "event_id": f"e{i}",
            "model_alias": "champion",
            "score": float(champion[i]),
            "decision": int(champion[i] > 0),
            "shadow_score": float(challenger[i]),
            "shadow_version": version,
        }
        for i in range(len(y))
    ]
    controller.on_predictions(preds[: len(y) // 2])
    controller.on_labels([{"event_id": f"e{i}", "label": int(y[i])} for i in range(len(y))])
    controller.on_predictions(preds[len(y) // 2 :])


def test_shadow_gate_promotes_to_canary_then_canary_promotes(tmp_path):
    controller, registry, redis = _controller(
        tmp_path,
        values={"histogram_quantile": 0.01, "scoring_errors": 0.0, "positive_decisions": 1.0},
    )
    y, champion = _scores(n=500, quality=1.0)
    challenger = np.random.default_rng(3).normal(size=500) + 4.0 * y
    _feed(controller, y, champion, challenger)
    record = controller.step()
    assert record["action"] == "promote_to_canary"
    assert redis.get("dg:rollout:cicids2017") == "canary"
    controller.on_predictions(
        [
            {
                "event_id": f"c{i}",
                "model_alias": "challenger",
                "score": 0.1,
                "decision": 0,
                "shadow_score": None,
                "shadow_version": None,
            }
            for i in range(150)
        ]
    )
    final = controller.step()
    assert final["action"] == "promote"
    assert registry.get_alias(NAME, CHAMPION).version == "2"
    assert registry.get_alias(NAME, PREVIOUS).version == "1"
    assert registry.get_alias(NAME, CHALLENGER) is None
    actions = [e["action"] for e in read_promotions(tmp_path / "promotions.jsonl")]
    assert actions == ["promote_to_canary", "promote"]


def test_shadow_gate_declines_equal_challenger(tmp_path):
    controller, registry, redis = _controller(tmp_path)
    y, champion = _scores(n=500)
    _feed(controller, y, champion, champion.copy())
    record = controller.step()
    assert record["action"] == "decline" and record["decision"] == "decline"
    assert registry.get_alias(NAME, CHALLENGER) is None
    assert redis.get("dg:rollout:cicids2017") == "champion_only"


def test_canary_guardrail_breach_rolls_back(tmp_path):
    controller, registry, redis = _controller(tmp_path, values={"histogram_quantile": 0.2})
    redis.set("dg:rollout:cicids2017", "canary")
    record = controller.step()
    assert record["action"] == "canary_rolled_back" and "p99_latency" in record["breaches"]
    assert registry.get_alias(NAME, CHAMPION).version == "1"
    assert registry.get_alias(NAME, CHALLENGER) is None
    assert redis.get("dg:rollout:cicids2017") == "champion_only"


def test_shadow_pairs_ignore_stale_challenger_versions(tmp_path):
    controller, *_ = _controller(tmp_path)
    y, champion = _scores(n=600)
    _feed(controller, y, champion, champion, version="old")
    assert controller.labeled == [] and controller.step() is None


def _canary_controller(tmp_path, values):
    controller, registry, redis = _controller(tmp_path, values=values, canary_min=100)
    redis.set("dg:rollout:cicids2017", "canary")
    controller.canary_events = 150
    return controller, registry, redis


def test_canary_never_promotes_without_guardrail_evidence(tmp_path):
    """Regression (M1): Prometheus down used to promote after canary_min_events."""
    controller, registry, redis = _canary_controller(tmp_path, values={})
    assert controller.step() is None  # first check: data missing, no promotion
    assert registry.get_alias(NAME, CHAMPION).version == "1"
    assert registry.get_alias(NAME, CHALLENGER) is not None
    assert redis.get("dg:rollout:cicids2017") == "canary"


def test_canary_rolls_back_after_repeated_missing_guardrail_data(tmp_path):
    controller, registry, redis = _canary_controller(tmp_path, values={})
    controller.config = {**controller.config, "guardrail_max_missing_checks": 2}
    controller.canary_events = 0
    records = [controller.step() for _ in range(4)]
    rolled = next(r for r in records if r)
    assert rolled["action"] == "canary_rolled_back"
    assert rolled["breaches"] == ["guardrail_data_unavailable"]
    assert registry.get_alias(NAME, CHALLENGER) is None
    assert redis.get("dg:rollout:cicids2017") == "champion_only"
