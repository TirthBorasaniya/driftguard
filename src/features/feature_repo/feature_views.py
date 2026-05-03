"""Feast feature view definitions: rolling 7-day card stats and per-category fraud rate."""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from feast import FeatureView, Field
from feast.types import Float32, Int64

from src.features.feature_repo.data_sources import card_stats_source, category_stats_source
from src.features.feature_repo.entities import card

# ============= View 1: Rolling 7-day card statistics =============

card_stats_7d = FeatureView(
    name="card_stats_7d",
    entities=[card],
    ttl=timedelta(days=30),
    schema=[
        Field(name="txn_count_7d", dtype=Int64),
        Field(name="amt_mean_7d", dtype=Float32),
        Field(name="amt_max_7d", dtype=Float32),
        Field(name="txn_velocity_7d", dtype=Float32),
    ],
    source=card_stats_source,
    online=True,
)

# ============= View 2: Per-category fraud rate =============
# category is the entity key for this view — reuse card entity join key

category_fraud_rate = FeatureView(
    name="category_fraud_rate",
    entities=[card],
    ttl=timedelta(days=30),
    schema=[
        Field(name="category_fraud_count", dtype=Int64),
        Field(name="category_txn_count", dtype=Int64),
        Field(name="category_fraud_rate", dtype=Float32),
    ],
    source=category_stats_source,
    online=True,
)
