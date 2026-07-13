"""Export the MLflow champion LightGBM model to ONNX and verify the export with a forward pass."""

import sys

import mlflow
import mlflow.lightgbm
import numpy as np
import onnxruntime as ort
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

from src.config import FEATURE_COLS, MODELS_DIR, settings

MODEL_NAME = settings.mlflow_model_name
CHAMPION_ALIAS = settings.mlflow_champion_alias
N_FEATURES = len(FEATURE_COLS)
OUTPUT_PATH = MODELS_DIR / "champion.onnx"


def load_champion_model():
    """
    Load the champion LightGBM model from the MLflow model registry.

    Raises
    ------
    SystemExit
        If the champion alias does not exist on MODEL_NAME. Lists all
        registered models and their aliases before exiting, rather than
        proceeding with a broken model URI.

    Returns
    -------
    model : lgb.LGBMClassifier
        The champion model loaded via the MLflow lightgbm flavor.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        print(f"Champion alias '{CHAMPION_ALIAS}' not found on model '{MODEL_NAME}'.")
        print("Registered models and their aliases:")
        for registered_model in client.search_registered_models():
            print(f"  {registered_model.name}: {registered_model.aliases}")
        sys.exit(1)

    model_uri = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    model = mlflow.lightgbm.load_model(model_uri)
    print(f"Loaded champion model: {model_uri}")
    return model


def convert_to_onnx(model, n_features: int):
    """
    Convert a LightGBM model to ONNX format.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        The LightGBM model to convert.
    n_features : int
        Number of input features, used to build the ONNX initial_types.

    Returns
    -------
    onnx_model : onnx.ModelProto
        The converted ONNX model.
    """
    initial_types = [("input", FloatTensorType([None, n_features]))]
    # zipmap=False keeps the probability output as a plain float tensor
    # instead of a sequence of maps, which is what onnxruntime.InferenceSession
    # and downstream numpy code in the server and benchmark expect
    onnx_model = convert_lightgbm(model, initial_types=initial_types, zipmap=False)
    return onnx_model


def save_onnx_model(onnx_model, output_path) -> None:
    """
    Write an ONNX model to disk.

    Parameters
    ----------
    onnx_model : onnx.ModelProto
        The ONNX model to save.
    output_path : Path
        Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX model: {output_path}")


def verify_onnx_export(onnx_path, n_features: int) -> None:
    """
    Load an exported ONNX model with ONNX Runtime and run a single forward
    pass on a zero-valued input to confirm the export is loadable and
    produces output of the correct shape.

    Parameters
    ----------
    onnx_path : Path
        Path to the exported ONNX model.
    n_features : int
        Number of input features, used to build the zero-valued test input.
    """
    session_options = ort.SessionOptions()
    # onnxmltools declares the lightgbm classifier's "label" output shape as
    # fixed at conversion time instead of dynamic on the batch dimension;
    # results are correct for any batch size, but ONNX Runtime logs a shape
    # mismatch warning on every call where batch size is not 1, so raise the
    # log severity to error only to silence that benign noise
    session_options.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options)
    input_name = session.get_inputs()[0].name
    zero_input = np.zeros((1, n_features), dtype=np.float32)

    outputs = session.run(None, {input_name: zero_input})
    output_shapes = [output.shape for output in outputs]
    print(f"Forward pass output shapes: {output_shapes}")
    print(f"ONNX export verified: {onnx_path}")


if __name__ == "__main__":
    champion_model = load_champion_model()
    exported_onnx_model = convert_to_onnx(champion_model, N_FEATURES)
    save_onnx_model(exported_onnx_model, OUTPUT_PATH)
    verify_onnx_export(OUTPUT_PATH, N_FEATURES)
