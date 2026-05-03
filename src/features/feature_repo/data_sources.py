"""Feast offline data sources pointing to pre-aggregated parquet files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from feast import FileSource

from src.config import CARD_STATS_FILE, CATEGORY_STATS_FILE

card_stats_source = FileSource(
    path=str(CARD_STATS_FILE),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

category_stats_source = FileSource(
    path=str(CATEGORY_STATS_FILE),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)
