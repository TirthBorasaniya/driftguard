"""Feast offline data source pointing to the per-src_ip network flow feature table."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from feast import FileSource

from src.config import NETWORK_FLOW_STATS_FILE

network_flow_source = FileSource(
    path=str(NETWORK_FLOW_STATS_FILE),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)
