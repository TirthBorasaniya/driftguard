"""Feast feature view: the ten network flow features keyed by source IP."""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from feast import FeatureView, Field
from feast.types import Float32

from src.features.engineering import FEATURE_COLS
from src.features.feature_repo.data_sources import network_flow_source
from src.features.feature_repo.entities import network_entity

# ============= Network Flow Feature View =============
# network flows are short-lived; a 1-hour TTL captures sufficient context for
# anomaly detection without stale feature contamination

network_flow_features = FeatureView(
    name="network_flow_features",
    entities=[network_entity],
    ttl=timedelta(hours=1),
    schema=[Field(name=col, dtype=Float32) for col in FEATURE_COLS],
    source=network_flow_source,
    online=True,
)
