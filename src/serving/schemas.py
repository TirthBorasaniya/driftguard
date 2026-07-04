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
