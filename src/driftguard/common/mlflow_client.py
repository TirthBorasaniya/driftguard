"""MLflow registry access by alias (champion, challenger, previous_champion), never stages."""

import contextlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# ============= Constants =============

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous_champion"
ALIASES = (CHAMPION, CHALLENGER, PREVIOUS)
MODEL_FILE, FEATURES_FILE, REFERENCE_FILE = "model.onnx", "features.json", "reference.parquet"


def model_name(dataset: str) -> str:
    """Registered model name per dataset."""
    return f"driftguard-{dataset}"


# ============= Registry interface =============


@dataclass(frozen=True)
class ModelRef:
    """A registered model version resolved from an alias."""

    name: str
    version: str
    tags: dict[str, str]
    source: str

    @property
    def threshold(self) -> float:
        """Decision threshold stored with the model version."""
        return float(self.tags["threshold"])


class Registry(Protocol):
    """Subset of registry operations used by serving, training, and the rollout controller."""

    def get_alias(self, name: str, alias: str) -> ModelRef | None:
        """Resolve an alias to a model version, or None."""
        ...

    def set_alias(self, name: str, alias: str, version: str) -> None:
        """Point an alias at a model version."""
        ...

    def delete_alias(self, name: str, alias: str) -> None:
        """Remove an alias if it exists."""
        ...

    def download(self, ref: ModelRef, dst: Path) -> Path:
        """Download a version's model directory into `dst`."""
        ...


class MlflowRegistry:
    """Registry backed by an MLflow tracking server."""

    def __init__(self, tracking_uri: str):
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    def get_alias(self, name: str, alias: str) -> ModelRef | None:
        """Resolve an alias, or None when it is not set."""
        from mlflow.exceptions import MlflowException

        try:
            version = self.client.get_model_version_by_alias(name, alias)
        except MlflowException:
            return None
        return ModelRef(name, str(version.version), dict(version.tags or {}), version.source)

    def set_alias(self, name: str, alias: str, version: str) -> None:
        """Point an alias at a version."""
        self.client.set_registered_model_alias(name, alias, str(version))

    def delete_alias(self, name: str, alias: str) -> None:
        """Remove an alias if present."""
        from mlflow.exceptions import MlflowException

        try:
            self.client.delete_registered_model_alias(name, alias)
        except MlflowException:
            return

    def download(self, ref: ModelRef, dst: Path) -> Path:
        """Download the version's model directory (ONNX, features, reference sample)."""
        import mlflow

        dst.mkdir(parents=True, exist_ok=True)
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{ref.name}/{ref.version}", dst_path=str(dst)
        )
        return Path(local)

    def register(
        self,
        dataset: str,
        artifacts: dict[str, bytes],
        tags: dict[str, Any],
        metrics: dict[str, float],
        params: dict[str, Any],
        experiment: str = "driftguard",
    ) -> str:
        """
        Log a run with the model directory and create a registered model version.

        Parameters
        ----------
        dataset : str
            Dataset name (model name suffix).
        artifacts : dict
            File name to bytes, written under the run's `model/` directory.
        tags : dict
            Version tags (must include `threshold`).
        metrics, params : dict
            Logged on the run.
        experiment : str
            Experiment name.

        Returns
        -------
        version : str
            New model version.
        """
        import mlflow
        from mlflow.exceptions import MlflowException

        name = model_name(dataset)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"{dataset}-{tags.get('trigger_id', 'manual')}") as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
            with tempfile.TemporaryDirectory() as tmp:
                model_dir = Path(tmp) / "model"
                model_dir.mkdir()
                for file_name, payload in artifacts.items():
                    (model_dir / file_name).write_bytes(payload)
                mlflow.log_artifacts(str(model_dir), artifact_path="model")
            source = f"{run.info.artifact_uri}/model"
        with contextlib.suppress(MlflowException):
            # already registered
            self.client.create_registered_model(name)
        version = self.client.create_model_version(name, source, run_id=run.info.run_id)
        for key, value in tags.items():
            self.client.set_model_version_tag(name, version.version, key, str(value))
        return str(version.version)


def features_json(names: list[str]) -> bytes:
    """Serialize the ordered model input feature names."""
    return json.dumps(names).encode()
