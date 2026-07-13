"""Benchmark native LightGBM inference against ONNX Runtime inference across batch sizes."""

import os
import platform
import time

import mlflow
import mlflow.lightgbm
import numpy as np
import onnxruntime as ort

from src.config import FEATURE_COLS, MODELS_DIR, PROJECT_ROOT, settings

BATCH_SIZES = [1, 8, 32, 128, 512, 1024]
N_WARMUP_ITERATIONS = 200
N_TIMED_ITERATIONS = 500
RANDOM_SEED = 42
N_FEATURES = len(FEATURE_COLS)
ONNX_MODEL_PATH = MODELS_DIR / "champion.onnx"
ONNX_INPUT_NAME = "input"
ONNX_OUTPUT_NAMES = ["label", "probabilities"]
OUTPUT_REPORT_PATH = PROJECT_ROOT / "BENCHMARK_RESULTS.md"

LIGHTGBM_PATH_LABEL = "LightGBM native"
ONNX_PATH_LABEL = "ONNX Runtime"


def load_native_model():
    """
    Load the champion LightGBM model from the MLflow model registry.

    Returns
    -------
    model : lgb.LGBMClassifier
        The champion model loaded via the MLflow lightgbm flavor.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_champion_alias}"
    model = mlflow.lightgbm.load_model(model_uri)
    return model


def load_onnx_session():
    """
    Load the exported ONNX champion model as an ONNX Runtime inference session.

    Returns
    -------
    session : onnxruntime.InferenceSession
    """
    session_options = ort.SessionOptions()
    # silences a benign ONNX Runtime shape-mismatch warning that fires on
    # every batch call with batch size greater than 1, caused by
    # onnxmltools declaring the classifier's "label" output shape as fixed
    # instead of dynamic on the batch dimension; results are unaffected
    session_options.log_severity_level = 3
    return ort.InferenceSession(str(ONNX_MODEL_PATH), sess_options=session_options)


def generate_batch(batch_size: int) -> np.ndarray:
    """
    Generate a reproducible random float32 feature batch.

    Parameters
    ----------
    batch_size : int
        Number of rows in the generated batch.

    Returns
    -------
    batch : np.ndarray
        Array of shape (batch_size, N_FEATURES), dtype float32.
    """
    np.random.seed(RANDOM_SEED)
    return np.random.rand(batch_size, N_FEATURES).astype(np.float32)


def run_native_inference(model, batch: np.ndarray) -> None:
    """Run one native LightGBM predict_proba call, discarding the result."""
    model.predict_proba(batch)


def run_onnx_inference(session, batch: np.ndarray) -> None:
    """Run one ONNX Runtime forward pass, discarding the result."""
    session.run(ONNX_OUTPUT_NAMES, {ONNX_INPUT_NAME: batch})


def time_inference_path(inference_fn, batch: np.ndarray) -> list[float]:
    """
    Warm up and time a single inference path against a fixed batch.

    Parameters
    ----------
    inference_fn : Callable[[np.ndarray], None]
        Bound inference call (native or ONNX) for a single batch.
    batch : np.ndarray
        Input batch used for every warmup and timed iteration.

    Returns
    -------
    elapsed_seconds_list : list of float
        Wall-clock elapsed seconds for each of the N_TIMED_ITERATIONS runs.
    """
    for _ in range(N_WARMUP_ITERATIONS):
        inference_fn(batch)

    elapsed_seconds_list = []
    for _ in range(N_TIMED_ITERATIONS):
        start_time = time.perf_counter()
        inference_fn(batch)
        elapsed_seconds_list.append(time.perf_counter() - start_time)

    return elapsed_seconds_list


def compute_stats(elapsed_seconds_list: list[float], batch_size: int) -> dict:
    """
    Compute p50 latency, p95 latency, and throughput from timed iterations.

    Parameters
    ----------
    elapsed_seconds_list : list of float
        Wall-clock elapsed seconds for each timed iteration.
    batch_size : int
        Number of records per iteration, used for throughput.

    Returns
    -------
    stats_dict : dict
        Keys: p50_ms, p95_ms, throughput_rps.
    """
    elapsed_ms_array = np.array(elapsed_seconds_list) * 1000.0
    total_elapsed_seconds = sum(elapsed_seconds_list)
    throughput_rps = (batch_size * N_TIMED_ITERATIONS) / total_elapsed_seconds

    return {
        "p50_ms": float(np.percentile(elapsed_ms_array, 50)),
        "p95_ms": float(np.percentile(elapsed_ms_array, 95)),
        "throughput_rps": throughput_rps,
    }


def print_result_row(result: dict) -> None:
    """Print a single benchmark result row in a fixed-width table format."""
    print(
        f"{result['path']:<16} | batch={result['batch_size']:>5} | "
        f"p50={result['p50_ms']:>8.4f} ms | p95={result['p95_ms']:>8.4f} ms | "
        f"throughput={result['throughput_rps']:>10.1f} rec/s"
    )


def run_benchmark() -> list[dict]:
    """
    Run the full native-vs-ONNX benchmark across all configured batch sizes.

    Returns
    -------
    results_list : list of dict
        One dict per (path, batch_size) combination with keys: path,
        batch_size, p50_ms, p95_ms, throughput_rps.
    """
    native_model = load_native_model()
    onnx_session = load_onnx_session()

    results_list = []

    print(f"{'Path':<16} | {'Batch':<11} | {'p50':<12} | {'p95':<12} | Throughput")
    print("-" * 80)

    for batch_size in BATCH_SIZES:
        batch = generate_batch(batch_size)

        native_elapsed_list = time_inference_path(
            lambda b: run_native_inference(native_model, b), batch
        )
        native_stats = compute_stats(native_elapsed_list, batch_size)
        native_result = {"path": LIGHTGBM_PATH_LABEL, "batch_size": batch_size, **native_stats}
        results_list.append(native_result)
        print_result_row(native_result)

        onnx_elapsed_list = time_inference_path(
            lambda b: run_onnx_inference(onnx_session, b), batch
        )
        onnx_stats = compute_stats(onnx_elapsed_list, batch_size)
        onnx_result = {"path": ONNX_PATH_LABEL, "batch_size": batch_size, **onnx_stats}
        results_list.append(onnx_result)
        print_result_row(onnx_result)

    return results_list


def get_hardware_description() -> str:
    """
    Describe the local hardware, preferring /proc/cpuinfo when available.

    Returns
    -------
    description : str
        Human-readable hardware description.
    """
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return f"{platform.machine()} ({platform.processor() or platform.system()}), {os.cpu_count()} logical CPUs"


def write_report(results_list: list[dict]) -> None:
    """
    Write BENCHMARK_RESULTS.md with the description, results table, and
    interpretation, using only the real measured values in results_list.

    Parameters
    ----------
    results_list : list of dict
        Benchmark results produced by run_benchmark.
    """
    hardware_description = get_hardware_description()
    python_version = platform.python_version()
    onnxruntime_version = ort.__version__
    import lightgbm

    lightgbm_version = lightgbm.__version__

    sorted_results = sorted(
        results_list,
        key=lambda r: (r["path"] != LIGHTGBM_PATH_LABEL, r["batch_size"]),
    )

    lightgbm_results = [r for r in results_list if r["path"] == LIGHTGBM_PATH_LABEL]
    onnx_results = [r for r in results_list if r["path"] == ONNX_PATH_LABEL]

    faster_summary_lines = []
    for lgbm_row, onnx_row in zip(lightgbm_results, onnx_results):
        batch_size = lgbm_row["batch_size"]
        if onnx_row["p50_ms"] < lgbm_row["p50_ms"]:
            factor = lgbm_row["p50_ms"] / onnx_row["p50_ms"]
            faster_summary_lines.append(f"batch {batch_size}: ONNX Runtime {factor:.2f}x faster")
        else:
            factor = onnx_row["p50_ms"] / lgbm_row["p50_ms"]
            faster_summary_lines.append(f"batch {batch_size}: LightGBM native {factor:.2f}x faster")

    lines = []
    lines.append("# ONNX Runtime vs. Native LightGBM Inference Benchmark")
    lines.append("")
    lines.append(
        f"Benchmarked native LightGBM `predict_proba` against an ONNX Runtime "
        f"`InferenceSession` running the exported champion model "
        f"(`models/champion.onnx`), on {hardware_description}, "
        f"Python {python_version}, onnxruntime {onnxruntime_version}, "
        f"lightgbm {lightgbm_version}. The model takes {N_FEATURES} numeric "
        f"features. Each (path, batch size) cell ran {N_WARMUP_ITERATIONS} "
        f"untimed warmup iterations followed by {N_TIMED_ITERATIONS} timed "
        f"iterations on a fixed, seeded (seed={RANDOM_SEED}) random input batch."
    )
    lines.append("")
    lines.append("| Path | Batch Size | p50 Latency (ms) | p95 Latency (ms) | Throughput (rec/s) |")
    lines.append("|---|---|---|---|---|")
    for result in sorted_results:
        lines.append(
            f"| {result['path']} | {result['batch_size']} | "
            f"{result['p50_ms']:.4f} | {result['p95_ms']:.4f} | "
            f"{result['throughput_rps']:.1f} |"
        )
    lines.append("")
    lines.append(
        "**Interpretation.** " + "; ".join(faster_summary_lines) + ". "
        "These are the real measured results from this run on this machine, "
        "not estimates."
    )
    lines.append("")

    with open(OUTPUT_REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote benchmark report: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    benchmark_results = run_benchmark()
    write_report(benchmark_results)
