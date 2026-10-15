"""Trigger log in SQLite (`orchestrator.db`) and the trigger decision (SPEC 6.1)."""

import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from driftguard.common.metrics import RETRAINING_RUNS, RETRAINING_TRIGGERS

# ============= Constants =============

SCHEMA = """
CREATE TABLE IF NOT EXISTS triggers (
    trigger_id TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    window_id TEXT NOT NULL,
    flow_run_id TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    detail TEXT
)
"""
ACTIVE_STATUSES = ("submitted", "running")
TERMINAL_MAP = {
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CRASHED": "failed",
    "CANCELLED": "cancelled",
    "CANCELLING": "cancelled",
}
PAUSE_KEY = "dg:retrain:paused:{dataset}"
ROLLOUT_KEY = "dg:rollout:{dataset}"
ACTIVE_ROLLOUT_MODES = ("shadow", "canary")


# ============= Store =============


class TriggerStore:
    """SQLite table of retraining triggers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def record(
        self,
        dataset: str,
        window_id: str,
        status: str,
        flow_run_id: str | None = None,
        detail: str = "",
    ) -> str:
        """Insert a trigger and return its id."""
        trigger_id = uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            "INSERT INTO triggers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (trigger_id, dataset, window_id, flow_run_id, status, now, now, detail),
        )
        self.conn.commit()
        return trigger_id

    def update(
        self,
        trigger_id: str,
        status: str,
        flow_run_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Update a trigger's status (and optionally run id and detail)."""
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            "UPDATE triggers SET status = ?, flow_run_id = COALESCE(?, flow_run_id), "
            "detail = COALESCE(?, detail), updated_at = ? WHERE trigger_id = ?",
            (status, flow_run_id, detail, now, trigger_id),
        )
        self.conn.commit()

    def active(self) -> list[dict[str, Any]]:
        """Triggers whose flow run has not reached a terminal state."""
        rows = self.conn.execute(
            "SELECT trigger_id, dataset, flow_run_id, status FROM triggers WHERE status IN (?, ?)",
            ACTIVE_STATUSES,
        ).fetchall()
        return [
            dict(zip(("trigger_id", "dataset", "flow_run_id", "status"), r, strict=True))
            for r in rows
        ]

    def all(self) -> list[dict[str, Any]]:
        """Every trigger, oldest first."""
        cursor = self.conn.execute("SELECT * FROM triggers ORDER BY created_at")
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


# ============= Decision =============


@dataclass
class TriggerOutcome:
    """Result of handling one drift event."""

    trigger_id: str
    status: str
    flow_run_id: str | None


def handle_drift_event(
    event: dict[str, Any],
    store: TriggerStore,
    redis_client,
    submit: Callable[[dict[str, Any]], str],
) -> TriggerOutcome:
    """
    Handle one drift event: skip while paused, otherwise submit a retraining run.

    Parameters
    ----------
    event : dict
        Decoded `drift.events` record.
    store : TriggerStore
        Trigger log.
    redis_client : redis.Redis
        Holds `dg:retrain:paused:{dataset}`.
    submit : callable
        Submits the deployment with parameters and returns the flow run id without waiting
        (`run_deployment(..., timeout=0)`).

    Returns
    -------
    outcome : TriggerOutcome
    """
    dataset = event["dataset"]
    paused = redis_client.get(PAUSE_KEY.format(dataset=dataset))
    if paused:
        trigger_id = store.record(dataset, event["window_id"], "skipped_paused", detail=paused)
        RETRAINING_TRIGGERS.labels(dataset=dataset, result="skipped_paused").inc()
        return TriggerOutcome(trigger_id, "skipped_paused", None)
    rollout_mode = redis_client.get(ROLLOUT_KEY.format(dataset=dataset))
    if rollout_mode in ACTIVE_ROLLOUT_MODES:
        # a new challenger would replace the alias under an in-flight shadow or canary stage
        trigger_id = store.record(
            dataset, event["window_id"], "skipped_active_rollout", detail=rollout_mode
        )
        RETRAINING_TRIGGERS.labels(dataset=dataset, result="skipped_active_rollout").inc()
        return TriggerOutcome(trigger_id, "skipped_active_rollout", None)
    trigger_id = store.record(dataset, event["window_id"], "pending")
    parameters = {
        "dataset": dataset,
        "trigger_id": trigger_id,
        "window_id": event["window_id"],
        "window_end_ms": int(event["signals"].get("window_end_ms", event["detected_at"])),
    }
    try:
        flow_run_id = submit(parameters)
    except Exception as exc:  # noqa: BLE001
        # Prefect API errors vary by transport; the trigger is recorded as failed, not lost
        store.update(trigger_id, "submit_failed", detail=str(exc)[:500])
        RETRAINING_TRIGGERS.labels(dataset=dataset, result="submit_failed").inc()
        return TriggerOutcome(trigger_id, "submit_failed", None)
    store.update(trigger_id, "submitted", flow_run_id=flow_run_id)
    RETRAINING_TRIGGERS.labels(dataset=dataset, result="submitted").inc()
    return TriggerOutcome(trigger_id, "submitted", flow_run_id)


RESULT_KEY = "dg:retrain:result:{trigger_id}"


def reconcile_runs(
    store: TriggerStore,
    get_state: Callable[[str], str | None],
    get_result: Callable[[str], str | None] = lambda _trigger_id: None,
) -> int:
    """
    Update active triggers from their flow run states.

    Parameters
    ----------
    store : TriggerStore
        Trigger log.
    get_state : callable
        Returns the Prefect state type name (`RUNNING`, `COMPLETED`, ...) for a flow run id.
    get_result : callable
        Returns the flow's own final status for a trigger (for example `registered`,
        `rejected_data_quality`), written by the flow to `dg:retrain:result:<trigger_id>`.

    Returns
    -------
    finished : int
        Number of triggers that reached a terminal state.
    """
    finished = 0
    for trigger in store.active():
        state = get_state(trigger["flow_run_id"]) if trigger["flow_run_id"] else None
        if state is None:
            continue
        if state in TERMINAL_MAP:
            status = TERMINAL_MAP[state]
            if status == "completed":
                status = get_result(trigger["trigger_id"]) or status
            store.update(trigger["trigger_id"], status)
            RETRAINING_RUNS.labels(dataset=trigger["dataset"], status=status).inc()
            finished += 1
        elif state == "RUNNING" and trigger["status"] != "running":
            store.update(trigger["trigger_id"], "running")
    return finished
