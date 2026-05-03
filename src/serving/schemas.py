"""Pydantic request and response models for the fraud detection API."""

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Raw transaction fields as received from the client or Kafka consumer."""

    cc_num: str
    merchant: str
    category: str
    amt: float = Field(gt=0)
    gender: str
    city: str
    state: str
    zip: str
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    merch_lat: float
    merch_long: float
    trans_date_trans_time: str


class PredictionResponse(BaseModel):
    """Standard fraud prediction response."""

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str
    timestamp: str


class FeatureContribution(BaseModel):
    """Single SHAP feature contribution."""

    feature: str
    shap_value: float


class ExplainResponse(PredictionResponse):
    """Fraud prediction with top-5 SHAP feature contributions."""

    top_features: list[FeatureContribution]


class HealthResponse(BaseModel):
    """API health and readiness status."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_predictions: int
