"""Prometheus metric definitions (SPEC 8) and a minimal metrics and health HTTP server."""

import re
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ============= Constants =============

PREFIX = "driftguard_"
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

# ============= Pipeline metrics =============

EVENTS_CONSUMED = Counter(
    f"{PREFIX}events_consumed_total", "Flow events consumed by the scoring consumer", ["dataset"]
)
EVENTS_DLQ = Counter(
    f"{PREFIX}events_dlq_total", "Events routed to the dead-letter queue", ["dataset", "error_type"]
)
DUPLICATES = Counter(
    f"{PREFIX}duplicates_total", "Events skipped by the idempotency check", ["dataset"]
)
CONSUMER_LAG = Gauge(
    f"{PREFIX}consumer_lag", "Consumer lag in messages per partition", ["dataset", "partition"]
)
SCORING_LATENCY = Histogram(
    f"{PREFIX}scoring_latency_seconds",
    "Model scoring latency per batch",
    ["dataset", "alias"],
    buckets=LATENCY_BUCKETS,
)
SCORING_ROUNDTRIP = Histogram(
    f"{PREFIX}scoring_roundtrip_seconds",
    "Consumer-side round trip of one scoring request (HTTP, serialization, inference)",
    ["dataset"],
    buckets=LATENCY_BUCKETS,
)
SCORING_ERRORS = Counter(
    f"{PREFIX}scoring_errors_total", "Scoring requests that failed", ["dataset", "alias"]
)
SCORING_REQUESTS = Counter(
    f"{PREFIX}scoring_requests_total", "Scoring requests served", ["dataset", "alias"]
)
POSITIVE_DECISIONS = Counter(
    f"{PREFIX}positive_decisions_total", "Events scored above threshold", ["dataset", "alias"]
)
SCORED_EVENTS = Counter(
    f"{PREFIX}scored_events_total", "Events scored by the serving router", ["dataset", "alias"]
)
SCORING_UNAVAILABLE = Counter(
    f"{PREFIX}scoring_unavailable_total",
    "Batches sent to the DLQ after scoring retries were exhausted",
    ["dataset"],
)
LATE_EVENTS = Counter(
    f"{PREFIX}late_events_total", "Events arriving after the allowed lateness", ["dataset"]
)
SHADOW_FAILURES = Counter(
    f"{PREFIX}shadow_failures_total", "Shadow scoring calls that failed", ["dataset"]
)
STREAM_FEATURE_SECONDS = Histogram(
    f"{PREFIX}stream_feature_seconds",
    "Time spent computing stream features per batch",
    ["dataset"],
    buckets=LATENCY_BUCKETS,
)
BATCH_SECONDS = Histogram(
    f"{PREFIX}consumer_batch_seconds",
    "End-to-end processing time per consumer batch",
    ["dataset"],
    buckets=LATENCY_BUCKETS,
)

# ============= Drift metrics =============

PSI_MAX = Gauge(f"{PREFIX}psi_max", "Maximum feature PSI in the last window", ["dataset"])
PSI_SHARE = Gauge(f"{PREFIX}psi_share", "Share of features with PSI above 0.2", ["dataset"])
DRIFT_SHARE = Gauge(f"{PREFIX}drift_share", "Evidently share of drifted columns", ["dataset"])
SCORE_PSI = Gauge(f"{PREFIX}score_psi", "PSI of the champion score distribution", ["dataset"])
ADWIN_DETECTIONS = Counter(
    f"{PREFIX}adwin_detections_total", "ADWIN change detections on the error stream", ["dataset"]
)
KSWIN_DETECTIONS = Counter(
    f"{PREFIX}kswin_detections_total", "KSWIN change detections on champion scores", ["dataset"]
)
DRIFT_EVENTS = Counter(
    f"{PREFIX}drift_events_total", "Drift events emitted", ["dataset", "severity"]
)
WINDOWS_CLOSED = Counter(f"{PREFIX}windows_closed_total", "Monitor windows closed", ["dataset"])
WINDOWS_FLAGGED = Counter(f"{PREFIX}windows_flagged_total", "Monitor windows flagged", ["dataset"])
DATA_QUALITY_FAILURES = Counter(
    f"{PREFIX}data_quality_failures_total", "Windows failing the GX suite", ["dataset"]
)
MONITOR_DECODE_FAILURES = Counter(
    f"{PREFIX}monitor_decode_failures_total", "Messages the monitor could not decode", ["dataset"]
)
RETRAINING_PAUSED = Gauge(
    f"{PREFIX}retraining_paused", "1 while retraining is paused for the dataset", ["dataset"]
)
LABELED_SINCE_PROMOTION = Gauge(
    f"{PREFIX}labeled_since_promotion", "Labeled events since the last promotion", ["dataset"]
)

# ============= Lifecycle metrics =============

RETRAINING_TRIGGERS = Counter(
    f"{PREFIX}retraining_triggers_total", "Retraining trigger outcomes", ["dataset", "result"]
)
RETRAINING_RUNS = Counter(
    f"{PREFIX}retraining_runs_total", "Retraining runs by final status", ["dataset", "status"]
)
PROMOTIONS = Counter(f"{PREFIX}promotions_total", "Promotion gate outcomes", ["dataset", "result"])
MODEL_INFO = Gauge(
    f"{PREFIX}model_info", "Loaded model version per alias", ["dataset", "alias", "version"]
)
SHADOW_AGREEMENT = Gauge(
    f"{PREFIX}shadow_agreement_ratio",
    "Share of shadow-scored events where champion and challenger decisions agree",
    ["dataset"],
)
ROLLOUT_MODE = Gauge(f"{PREFIX}rollout_mode", "1 for the active rollout mode", ["dataset", "mode"])


# ============= HTTP server =============


def _make_handler(
    registry: CollectorRegistry, health_check: Callable[[], bool]
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class serving `/metrics` and `/healthz`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            """Serve `/metrics` and `/healthz`."""
            if self.path.startswith("/metrics"):
                body = generate_latest(registry)
                self._reply(200, body, CONTENT_TYPE_LATEST)
            elif self.path.startswith("/healthz"):
                healthy = health_check()
                self._reply(200 if healthy else 503, b"ok" if healthy else b"unhealthy")
            else:
                self._reply(404, b"not found")

        def _reply(self, status: int, body: bytes, content_type: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: object) -> None:
            """Silence the default per-request logging."""
            return

    return Handler


def start_metrics_server(
    port: int,
    health_check: Callable[[], bool] = lambda: True,
    registry: CollectorRegistry = REGISTRY,
    host: str = "0.0.0.0",  # noqa: S104
) -> ThreadingHTTPServer:
    """
    Start a background HTTP server exposing `/metrics` and `/healthz`.

    Parameters
    ----------
    port : int
        TCP port to bind.
    health_check : callable
        Returns True while the service is healthy.
    registry : CollectorRegistry
        Prometheus registry to expose.
    host : str
        Bind address; containers bind all interfaces, Compose publishes on loopback only.

    Returns
    -------
    server : ThreadingHTTPServer
        Running server; call `shutdown()` to stop it.
    """
    server = ThreadingHTTPServer((host, port), _make_handler(registry, health_check))
    thread = threading.Thread(target=server.serve_forever, name="metrics-server", daemon=True)
    thread.start()
    return server


# ============= Scraping =============

METRIC_LINE = re.compile(r"^(?P<name>[a-z_:]+)(\{(?P<labels>[^}]*)\})? (?P<value>[0-9.eE+-]+|NaN)$")


def scrape_metric(url: str, name: str, **labels: str) -> float:
    """
    Sum one metric from a `/metrics` endpoint, filtered by label values.

    Parameters
    ----------
    url : str
        Metrics endpoint.
    name : str
        Sample name (for counters include `_total`).
    labels : dict
        Required label values.

    Returns
    -------
    total : float
        Sum over matching samples; 0.0 when none match.
    """
    import httpx

    total = 0.0
    for line in httpx.get(url, timeout=5).text.splitlines():
        match = METRIC_LINE.match(line)
        if not match or match.group("name") != name or match.group("value") == "NaN":
            continue
        found = dict(re.findall(r'(\w+)="([^"]*)"', match.group("labels") or ""))
        if all(found.get(k) == v for k, v in labels.items()):
            total += float(match.group("value"))
    return total
