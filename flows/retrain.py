"""Prefect flow `retrain`, deployed as `retrain/retrain-on-drift` (SPEC 6.1, 6.2)."""

from prefect import flow, get_run_logger

from driftguard.common.settings import get_settings
from driftguard.training.deps import build_retrain_deps
from driftguard.training.flow import run_retraining

# ============= Flow =============


@flow(name="retrain", log_prints=False)
def retrain_on_drift(dataset: str, trigger_id: str, window_id: str, window_end_ms: int) -> dict:
    """
    Retrain on the newest labeled window and register a challenger.

    Parameters
    ----------
    dataset : str
        Dataset name.
    trigger_id : str
        Orchestrator trigger id (MLflow tag).
    window_id : str
        Monitor window that confirmed drift.
    window_end_ms : int
        Event time bounding the training window.

    Returns
    -------
    summary : dict
        Final status (`registered`, `rejected_data_quality`, `rejected_insufficient_data`).
    """
    logger = get_run_logger()
    params = {
        "dataset": dataset,
        "trigger_id": trigger_id,
        "window_id": window_id,
        "window_end_ms": window_end_ms,
    }
    summary = run_retraining(params, build_retrain_deps(dataset, get_settings()))
    logger.info("retrain finished: %s", summary["status"])
    return summary
