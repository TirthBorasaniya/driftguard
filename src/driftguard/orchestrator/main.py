"""Orchestrator: consumes drift events and submits retraining runs without waiting (SPEC 6.1)."""

import asyncio
import signal
import threading
import time

from driftguard.common.config import load_config
from driftguard.common.logging import configure_logging, get_logger
from driftguard.common.metrics import start_metrics_server
from driftguard.common.settings import get_settings
from driftguard.orchestrator.triggers import (
    RESULT_KEY,
    TriggerStore,
    handle_drift_event,
    reconcile_runs,
)

# ============= Constants =============

logger = get_logger("orchestrator")


# ============= Prefect adapters =============


def submit_deployment(deployment: str, parameters: dict) -> str:
    """Create a flow run of the deployment and return immediately (`timeout=0`)."""
    from prefect.deployments import run_deployment

    flow_run = run_deployment(name=deployment, parameters=parameters, timeout=0, as_subflow=False)
    return str(flow_run.id)


def flow_run_state(flow_run_id: str) -> str | None:
    """Return the state type name of a flow run, or None if it cannot be read."""
    from prefect.client.orchestration import get_client

    async def _read() -> str | None:
        async with get_client() as client:
            run = await client.read_flow_run(flow_run_id)
            return run.state.type.value if run.state else None

    try:
        return asyncio.run(_read())
    except Exception as exc:  # noqa: BLE001
        # the Prefect API may be briefly unavailable; the next reconcile pass retries
        logger.warning("flow_run_state_unavailable", flow_run_id=flow_run_id, error=str(exc))
        return None


# ============= Service =============


class OrchestratorService:
    """Consumes `drift.events`, records triggers, submits runs, and reconciles run states."""

    def __init__(self):
        from confluent_kafka import Consumer
        from redis import Redis

        from driftguard.common.kafka import consumer_config
        from driftguard.data.schemas import topic_name
        from driftguard.data.serde import TopicSerde, registry_client

        self.settings = get_settings()
        self.config = load_config(self.settings.config_path("retrain.yaml"))
        self.store = TriggerStore(self.settings.orchestrator_db_path)
        self.redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        self.consumer = Consumer(
            consumer_config(
                self.settings.kafka_bootstrap_servers, "driftguard-orchestrator", "orchestrator"
            )
        )
        self.consumer.subscribe([topic_name("drift")])
        self.serde = TopicSerde(registry_client(self.settings.schema_registry_url), "drift")
        self._running = True

    def healthy(self) -> bool:
        """Liveness for `/healthz`."""
        return self._running

    def stop(self, *_args: object) -> None:
        """Stop after the current poll."""
        self._running = False

    def _submit(self, parameters: dict) -> str:
        return submit_deployment(self.config["deployment"], parameters)

    def run(self) -> None:
        """Handle drift events and reconcile run states every `status_poll_s` seconds."""
        last_reconcile = 0.0
        while self._running:
            message = self.consumer.poll(1.0)
            if message is not None and not message.error():
                event = self.serde.deserialize(message.value())
                outcome = handle_drift_event(event, self.store, self.redis, self._submit)
                logger.info(
                    "drift_event_handled",
                    window_id=event["window_id"],
                    trigger_id=outcome.trigger_id,
                    status=outcome.status,
                    flow_run_id=outcome.flow_run_id,
                )
                self.consumer.commit(message=message, asynchronous=False)
            if time.monotonic() - last_reconcile >= float(self.config["status_poll_s"]):
                reconcile_runs(
                    self.store,
                    flow_run_state,
                    lambda tid: self.redis.get(RESULT_KEY.format(trigger_id=tid)),
                )
                last_reconcile = time.monotonic()
        self.consumer.close()


def main() -> None:
    """Entry point `python -m driftguard.orchestrator.main`: trigger loop and rollout controller."""
    settings = get_settings()
    configure_logging("orchestrator", settings.log_level)
    service = OrchestratorService()
    from driftguard.orchestrator.rollout import RolloutService

    rollout = RolloutService(settings.dataset)
    signal.signal(signal.SIGTERM, lambda *_: (service.stop(), rollout.stop()))
    signal.signal(signal.SIGINT, lambda *_: (service.stop(), rollout.stop()))
    start_metrics_server(settings.metrics_port, lambda: service.healthy() and rollout.healthy())
    threading.Thread(target=rollout.run, name="rollout", daemon=True).start()
    service.run()


if __name__ == "__main__":
    main()
