"""ONNX export of LightGBM models with a native-versus-ONNX parity check (SPEC 6.2 step 4)."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

# ============= Constants =============

INPUT_NAME = "input"
BATCH_AXIS = "batch"


# ============= Export =============


def set_dynamic_batch(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Declare the leading dimension of every graph input and output as the dynamic `batch` axis.

    The LightGBM converter declares the `label` output with a fixed shape of 1, so ONNX Runtime
    warns on every batch of another size. The model is modified in place.

    Parameters
    ----------
    model : onnx.ModelProto
        Converted model.

    Returns
    -------
    model : onnx.ModelProto
        The same model with a dynamic batch dimension on inputs and outputs.
    """
    for value in [*model.graph.input, *model.graph.output]:
        dims = value.type.tensor_type.shape.dim
        if len(dims):
            dims[0].Clear()
            dims[0].dim_param = BATCH_AXIS
    return model


def to_onnx_bytes(
    model: lgb.LGBMClassifier, n_features: int, split: int | None = 100, target_opset: int = 15
) -> bytes:
    """
    Convert a fitted LightGBM classifier to serialized ONNX.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Fitted model.
    n_features : int
        Input width.
    split : int or None
        Trees per TreeEnsemble node; smaller chunks reduce float32 summation error.
    target_opset : int
        ONNX opset.

    Returns
    -------
    payload : bytes
        Serialized ONNX model with outputs `label` and `probabilities` (no ZipMap) and a
        dynamic batch dimension on the input and both outputs.
    """
    onnx_model = convert_lightgbm(
        model,
        initial_types=[(INPUT_NAME, FloatTensorType([None, n_features]))],
        zipmap=False,
        target_opset=target_opset,
        split=split,
    )
    return set_dynamic_batch(onnx_model).SerializeToString()


def onnx_session(payload: bytes | str | Path, threads: int = 1) -> ort.InferenceSession:
    """Create an ONNX Runtime CPU session from bytes or a file path."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    source = payload if isinstance(payload, bytes) else str(payload)
    return ort.InferenceSession(source, sess_options=options, providers=["CPUExecutionProvider"])


def onnx_scores(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities from an ONNX session, shape (n,)."""
    outputs = session.run(None, {INPUT_NAME: np.asarray(x, dtype=np.float32)})
    probabilities = outputs[1]
    return np.asarray(probabilities)[:, 1].astype(np.float64)


def parity_check(
    model: lgb.LGBMClassifier,
    session: ort.InferenceSession,
    x: np.ndarray,
    n_rows: int = 1000,
    max_abs_diff: float = 1e-5,
    seed: int = 0,
) -> float:
    """
    Compare native and ONNX scores on up to `n_rows` rows.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Fitted LightGBM classifier.
    session : ort.InferenceSession
        ONNX Runtime session of the exported model.
    x : np.ndarray
        Feature matrix to sample rows from.
    n_rows : int
        Maximum rows compared.
    max_abs_diff : float
        Largest tolerated absolute difference.
    seed : int
        Seed for the row sample.

    Returns
    -------
    observed : float
        Maximum absolute difference observed.

    Raises
    ------
    ValueError
        If the difference exceeds `max_abs_diff`; the tolerance is never relaxed.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(n_rows, len(x)), replace=False)
    sample = np.asarray(x, dtype=np.float32)[idx]
    native = model.predict_proba(sample)[:, 1]
    observed = float(np.max(np.abs(native - onnx_scores(session, sample))))
    if observed > max_abs_diff:
        raise ValueError(f"ONNX parity failed: max abs diff {observed:.3e} > {max_abs_diff:.1e}")
    return observed
