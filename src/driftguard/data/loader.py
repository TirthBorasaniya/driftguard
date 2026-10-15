"""Dataset-agnostic loading of the stream-ordered event frame used by the producer and evaluations."""

import pandas as pd

from driftguard.common.config import LoadedConfig
from driftguard.data.synthetic import synthetic_flows

# ============= Loading =============


def load_stream_frame(
    dataset: str,
    source: str = "dataset",
    producer_config: LoadedConfig | None = None,
    o_run_from_scratch: bool = False,
) -> pd.DataFrame:
    """
    Return the event frame to replay, in event-time order.

    Parameters
    ----------
    dataset : str
        `cicids2017` or `kyoto`.
    source : str
        `dataset` for the real adapter output or `synthetic` for generated flows.
    producer_config : LoadedConfig or None
        Producer config; supplies the synthetic parameters.
    o_run_from_scratch : bool
        Rebuild adapter caches.

    Returns
    -------
    frame : pd.DataFrame
        Identifiers, event time, optional IPs, features, `label`, `label_family`.
    """
    if source == "synthetic":
        params = (producer_config.get("synthetic", {}) if producer_config else {}) or {}
        seed = producer_config.get("seed", 42) if producer_config else 42
        return synthetic_flows(
            dataset,
            n_events=int(params.get("n_events", 10_000)),
            attack_rate=float(params.get("attack_rate", 0.05)),
            seed=int(seed),
            start_time_ms=int(params.get("start_time_ms", 1_499_083_200_000)),
            step_ms=int(params.get("step_ms", 5)),
        )
    if dataset == "cicids2017":
        from driftguard.data.cicids2017 import load_dataset

        frame, _ = load_dataset(o_run_from_scratch=o_run_from_scratch)
        return frame
    if dataset == "kyoto":
        from driftguard.data.kyoto import load_stream

        return load_stream(o_run_from_scratch=o_run_from_scratch)
    raise ValueError(f"unknown dataset {dataset!r}")
