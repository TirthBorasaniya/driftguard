"""Standalone FastAPI application serving predictions from the exported ONNX champion model."""

import time
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.config import FEATURE_COLS, MODELS_DIR

ONNX_MODEL_PATH = MODELS_DIR / "champion.onnx"
N_FEATURES = len(FEATURE_COLS)
INPUT_NAME = "input"
LABEL_OUTPUT_NAME = "label"
PROBABILITY_OUTPUT_NAME = "probabilities"


class PredictRequest(BaseModel):
    """Request body for /predict: a batch of feature vectors."""

    features: list[list[float]]

    @field_validator("features")
    @classmethod
    def validate_feature_vector_length(cls, value: list[list[float]]) -> list[list[float]]:
        """Reject any feature vector that does not have exactly N_FEATURES values."""
        for index, feature_vector in enumerate(value):
            if len(feature_vector) != N_FEATURES:
                raise ValueError(
                    f"Feature vector at index {index} has {len(feature_vector)} values, "
                    f"expected exactly {N_FEATURES}."
                )
        return value


class PredictResponse(BaseModel):
    """Response body for /predict."""

    predictions: list[int]
    probabilities: list[list[float]]
    latency_ms: float
    batch_size: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX Runtime inference session once at startup and share it via app.state."""
    session_options = ort.SessionOptions()
    # silences a benign ONNX Runtime shape-mismatch warning that fires on
    # every batch request with batch size greater than 1, caused by
    # onnxmltools declaring the classifier's "label" output shape as fixed
    # instead of dynamic on the batch dimension; results are unaffected
    session_options.log_severity_level = 3
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), sess_options=session_options)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"ONNX Runtime session loaded: {ONNX_MODEL_PATH}")
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")

    app.state.session = session
    yield


app = FastAPI(title="DriftGuard ONNX Inference Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Report server status, serving backend, and expected feature count."""
    return {"status": "ok", "model": "onnx", "n_features": N_FEATURES}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Run inference on a batch of feature vectors using ONNX Runtime.

    Parameters
    ----------
    request : PredictRequest
        Batch of feature vectors, each of length N_FEATURES.

    Returns
    -------
    response : PredictResponse
        Predictions, probabilities, measured latency, and batch size.
    """
    session: ort.InferenceSession = app.state.session

    if not request.features:
        raise HTTPException(status_code=422, detail="features must not be empty.")

    input_array = np.asarray(request.features, dtype=np.float32)

    start_time = time.perf_counter()
    outputs = session.run(
        [LABEL_OUTPUT_NAME, PROBABILITY_OUTPUT_NAME],
        {INPUT_NAME: input_array},
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    labels, probabilities = outputs

    return PredictResponse(
        predictions=labels.tolist(),
        probabilities=probabilities.tolist(),
        latency_ms=elapsed_ms,
        batch_size=input_array.shape[0],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
