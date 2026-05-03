"""Feast entity definition for the fraud detection pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from feast import Entity

card = Entity(
    name="cc_num",
    join_keys=["cc_num"],
    description="Credit card number — primary entity for transaction features.",
)
