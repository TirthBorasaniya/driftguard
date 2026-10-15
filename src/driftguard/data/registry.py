"""Dataset registry: config paths, feature lists, and adapter dispatch."""

from functools import lru_cache
from pathlib import Path

from driftguard.common.config import LoadedConfig, load_config
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

DATASETS = ("cicids2017", "kyoto")
CONFIG_DIR = REPO_ROOT / "configs" / "data"


# ============= Registry =============


def dataset_config_path(dataset: str) -> Path:
    """Return the adapter config path for a dataset."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    return CONFIG_DIR / f"{dataset}.yaml"


@lru_cache(maxsize=8)
def dataset_config(dataset: str) -> LoadedConfig:
    """Load and cache a dataset adapter config."""
    return load_config(dataset_config_path(dataset))


def feature_specs(dataset: str, source: str | None = None) -> list[dict]:
    """
    Return the feature specs available for a dataset and source.

    Parameters
    ----------
    dataset : str
        Dataset name.
    source : str or None
        CICIDS2017 source (`improved` or `original`); defaults to the configured source.
        Features absent from the selected source are dropped.

    Returns
    -------
    specs : list of dict
        Each dict has `name`, the source column, `min`, and `max`.
    """
    config = dataset_config(dataset)
    specs = config["features"]
    if dataset != "cicids2017":
        return list(specs)
    source = source or config["source"]
    return [spec for spec in specs if spec.get(source)]


def feature_names(dataset: str, source: str | None = None) -> list[str]:
    """Return the ordered model feature names for a dataset."""
    return [spec["name"] for spec in feature_specs(dataset, source)]


def stream_features_enabled(dataset: str) -> bool:
    """Return True if stream-computed behavioral features are enabled for the dataset."""
    return bool(dataset_config(dataset).get("stream_features", {}).get("enabled", False))
