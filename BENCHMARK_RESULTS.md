# ONNX Runtime vs. Native LightGBM Inference Benchmark

Benchmarked native LightGBM `predict_proba` against an ONNX Runtime `InferenceSession` running the exported champion model (`models/champion.onnx`), on arm64 (arm), 10 logical CPUs, Python 3.11.9, onnxruntime 1.27.0, lightgbm 4.3.0. The model takes 10 numeric features. Each (path, batch size) cell ran 200 untimed warmup iterations followed by 500 timed iterations on a fixed, seeded (seed=42) random input batch.

| Path | Batch Size | p50 Latency (ms) | p95 Latency (ms) | Throughput (rec/s) |
|---|---|---|---|---|
| LightGBM native | 1 | 0.1583 | 0.2872 | 5566.4 |
| LightGBM native | 8 | 0.1549 | 0.2108 | 49720.7 |
| LightGBM native | 32 | 0.1603 | 0.3891 | 169060.9 |
| LightGBM native | 128 | 0.1609 | 0.2159 | 757463.7 |
| LightGBM native | 512 | 0.1677 | 0.2148 | 2887496.6 |
| LightGBM native | 1024 | 0.1937 | 0.2551 | 5077729.9 |
| ONNX Runtime | 1 | 0.0030 | 0.0032 | 336416.3 |
| ONNX Runtime | 8 | 0.0030 | 0.0038 | 2605629.9 |
| ONNX Runtime | 32 | 0.0038 | 0.0041 | 8012210.2 |
| ONNX Runtime | 128 | 0.0048 | 0.0059 | 24839048.9 |
| ONNX Runtime | 512 | 0.0100 | 0.0120 | 48659666.2 |
| ONNX Runtime | 1024 | 0.0153 | 0.0195 | 62378481.8 |

**Interpretation.** batch 1: ONNX Runtime 53.53x faster; batch 8: ONNX Runtime 51.63x faster; batch 32: ONNX Runtime 41.82x faster; batch 128: ONNX Runtime 33.30x faster; batch 512: ONNX Runtime 16.84x faster; batch 1024: ONNX Runtime 12.63x faster. These are the real measured results from this run on this machine, not estimates.
