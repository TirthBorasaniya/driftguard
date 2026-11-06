"""Tests for the ONNX export: dynamic batch dimension and warning-free scoring of any batch size."""

import lightgbm as lgb
import numpy as np
import onnx
import pytest

from driftguard.training.export import (
    BATCH_AXIS,
    INPUT_NAME,
    onnx_scores,
    onnx_session,
    to_onnx_bytes,
)

pytestmark = pytest.mark.unit

N_FEATURES = 12
BATCH_SIZES = (1, 1000, 4000)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, N_FEATURES)).astype(np.float32)
    y = (x[:, 0] + rng.normal(size=2000) > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=30, verbose=-1, n_jobs=1).fit(x, y)
    return model, to_onnx_bytes(model, N_FEATURES)


def test_every_input_and_output_has_a_dynamic_batch_dimension(fitted):
    graph = onnx.load_from_string(fitted[1]).graph
    assert [v.name for v in graph.output] == ["label", "probabilities"]
    for value in [*graph.input, *graph.output]:
        first = value.type.tensor_type.shape.dim[0]
        assert first.dim_param == BATCH_AXIS and first.dim_value == 0, value.name
    assert graph.input[0].name == INPUT_NAME


def test_scoring_batches_of_any_size_emits_no_shape_warning(fitted, capfd):
    """Regression: the `label` output was declared with shape 1, so batches of 1000 and 4000 rows
    made ONNX Runtime log `Expected shape from model of {1} does not match actual shape`."""
    model, payload = fitted
    session = onnx_session(payload)
    rng = np.random.default_rng(1)
    capfd.readouterr()
    for n in BATCH_SIZES:
        x = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
        scores = onnx_scores(session, x)
        assert scores.shape == (n,)
        assert np.max(np.abs(scores - model.predict_proba(x)[:, 1])) < 1e-5
    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "VerifyOutputSizes" not in output and "does not match actual shape" not in output
