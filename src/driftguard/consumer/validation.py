"""Schema-level validation of decoded flow events (SPEC 4.1 step 2)."""

import math
from dataclasses import dataclass
from typing import Any

from driftguard.data.registry import feature_specs
from driftguard.data.schemas import DATASETS_WITH_IPS

# ============= Constants =============

BASE_REQUIRED = ("event_id", "dataset", "source_file", "row_index", "event_time")


# ============= Validator =============


@dataclass(frozen=True)
class ValidationError(Exception):
    """A flow event failed schema-level validation."""

    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


class FlowValidator:
    """
    Checks required fields, finite numbers, and the physical bounds in the dataset config.

    Parameters
    ----------
    dataset : str
        Dataset whose feature specs define the checks.
    """

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.specs = [
            (spec["name"], spec.get("min"), spec.get("max")) for spec in feature_specs(dataset)
        ]
        self.required = BASE_REQUIRED + (
            ("src_ip", "dst_ip") if dataset in DATASETS_WITH_IPS else ()
        )

    def validate(self, record: dict[str, Any]) -> None:
        """
        Validate one decoded record.

        Raises
        ------
        ValidationError
            On the first failed check.
        """
        for field in self.required:
            value = record.get(field)
            if value is None or value == "":
                raise ValidationError(field, "missing")
        if record["dataset"] != self.dataset:
            raise ValidationError("dataset", f"expected {self.dataset}")
        for name, low, high in self.specs:
            value = record.get(name)
            if value is None:
                raise ValidationError(name, "missing")
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise ValidationError(name, "not a finite number")
            if low is not None and value < low:
                raise ValidationError(name, f"below {low}")
            if high is not None and value > high:
                raise ValidationError(name, f"above {high}")
