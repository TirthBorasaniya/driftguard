"""Feast entity definition for the network telemetry anomaly detection pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from feast import Entity

network_entity = Entity(
    name="src_ip",
    join_keys=["src_ip"],
    description="Source IP address — primary entity for network flow features.",
)
