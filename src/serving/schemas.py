"""Pydantic request and response models for the network anomaly detection API."""

from pydantic import BaseModel


class NetworkFlowRequest(BaseModel):
    """Raw network flow features as received from the client or Kafka consumer."""

    flow_duration: float
    flow_bytes_per_sec: float
    flow_packets_per_sec: float
    total_fwd_packets: float
    total_bwd_packets: float
    packet_length_mean: float
    packet_length_std: float
    flow_iat_mean: float
    syn_flag_count: float
    src_ip: str | None = None
    flow_id: str | None = None


class PredictionResponse(BaseModel):
    """Standard network anomaly prediction response."""

    event_id: str
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    model_version: str
    timestamp: str


class FeatureContribution(BaseModel):
    """Single SHAP feature contribution."""

    feature: str
    shap_value: float


class ExplainResponse(PredictionResponse):
    """Network anomaly prediction with top-5 SHAP feature contributions."""

    top_features: list[FeatureContribution]


class HealthResponse(BaseModel):
    """API health and readiness status."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_predictions: int


class RecentPredictionItem(BaseModel):
    """
    One row of the public read-only prediction feed.

    Deliberately excludes two stored columns that can carry network
    identifiers:

    - features_json, which persists the submitted src_ip
    - event_id, which defaults to the caller-supplied flow_id, and whose
      CICIDS2017 form is the full five-tuple
      (src_ip-dst_ip-src_port-dst_port-protocol)

    Also excludes the SQLite rowid, which is internal bookkeeping. What
    remains is the model's decision and the model version that made it,
    which is the part worth demonstrating.
    """

    anomaly_score: float
    is_anomaly: bool
    threshold: float
    model_version: str
    timestamp: str


class PredictionStatsResponse(BaseModel):
    """
    Aggregate-only prediction statistics.

    Reviewed for operational disclosure: these four aggregates contain no
    identifiers, no per-event rows, no filesystem paths, and no
    infrastructure detail, so all are retained for the demo. The response is
    typed explicitly so a schema change cannot silently widen the payload.
    """

    total_predictions: int
    total_anomaly: int = 0
    anomaly_rate: float = 0.0
    avg_score: float | None = None
