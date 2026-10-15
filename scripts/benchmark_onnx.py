"""Benchmark native LightGBM against ONNX Runtime: writes results/onnx_latency.json."""

import argparse
import time
from pathlib import Path

import numpy as np

from driftguard.common.config import load_config
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT
from driftguard.data.cicids2017 import load_dataset
from driftguard.data.registry import feature_names
from driftguard.training.export import onnx_scores, onnx_session, to_onnx_bytes
from driftguard.training.train import train_model

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "onnx_latency.yaml"


# ============= Timing =============


def time_calls(fn, x: np.ndarray, warmup: int, iterations: int) -> dict[str, float]:
    """p50 and p95 latency in milliseconds and throughput in rows per second."""
    for _ in range(warmup):
        fn(x)
    samples = np.empty(iterations)
    for i in range(iterations):
        start = time.perf_counter()
        fn(x)
        samples[i] = time.perf_counter() - start
    p50 = float(np.percentile(samples, 50))
    return {
        "p50_ms": p50 * 1000,
        "p95_ms": float(np.percentile(samples, 95)) * 1000,
        "throughput_rows_per_s": len(x) / p50,
    }


def main() -> None:
    """Train the champion configuration, export it, and time both inference paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config).data
    model_cfg = load_config(REPO_ROOT / config["model_config"]).data
    features = feature_names(config["dataset"])
    frame, _ = load_dataset(days=list(config["train_days"]))
    x = frame[features].to_numpy(np.float32)
    model, _ = train_model(
        x, frame["label"].to_numpy(), model_cfg["lightgbm"], int(model_cfg["seed"])
    )
    session = onnx_session(
        to_onnx_bytes(
            model,
            len(features),
            int(model_cfg["onnx"]["split"]),
            int(model_cfg["onnx"]["target_opset"]),
        ),
        threads=int(config["onnx_threads"]),
    )
    rng = np.random.default_rng(int(config["input_seed"]))
    rows = []
    for batch in config["batch_sizes"]:
        sample = x[rng.choice(len(x), size=batch, replace=False)]
        native = time_calls(
            lambda b: model.predict_proba(b)[:, 1],
            sample,
            int(config["warmup_iterations"]),
            int(config["timed_iterations"]),
        )
        onnx = time_calls(
            lambda b: onnx_scores(session, b),
            sample,
            int(config["warmup_iterations"]),
            int(config["timed_iterations"]),
        )
        rows.append(
            {
                "batch_size": batch,
                "native": native,
                "onnx": onnx,
                "speedup_p50": native["p50_ms"] / onnx["p50_ms"],
            }
        )
        print(f"batch {batch}: native {native['p50_ms']:.4f} ms, onnx {onnx['p50_ms']:.4f} ms")
    metrics = {
        "n_features": len(features),
        "n_estimators": model_cfg["lightgbm"]["n_estimators"],
        "rows": rows,
    }
    notes = (
        "Same LightGBM configuration as the champion, trained on Monday and Tuesday. "
        "Single-threaded ONNX Runtime session; native path is LGBMClassifier.predict_proba."
    )
    print(f"Wrote {write_results('onnx_latency', metrics, args.config, notes=notes)}")


if __name__ == "__main__":
    main()
