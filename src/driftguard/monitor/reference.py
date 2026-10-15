"""Reference window of the current champion, loaded from its MLflow artifacts (SPEC 5.1)."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.common.mlflow_client import (
    CHAMPION,
    FEATURES_FILE,
    REFERENCE_FILE,
    ModelRef,
    Registry,
    model_name,
)
from driftguard.monitor.detectors import EvidentlyDriftShare, ScorePsi
from driftguard.monitor.psi import PsiProfile

# ============= Reference =============


@dataclass
class Reference:
    """Everything the monitor compares windows against."""

    version: str
    features: list[str]
    frame: pd.DataFrame
    psi_profile: PsiProfile
    score_psi: ScorePsi
    evidently: EvidentlyDriftShare


def build_reference(
    version: str, frame: pd.DataFrame, features: list[str], n_bins: int, evidently_rows: int
) -> Reference:
    """
    Fit PSI bins, score bins, and the Evidently reference from a reference frame.

    Parameters
    ----------
    version : str
        Champion version the reference belongs to.
    frame : pd.DataFrame
        Reference rows: the feature columns and `score` (champion scores on the same rows).
    features : list of str
        Monitored feature columns.
    n_bins : int
        PSI bins.
    evidently_rows : int
        Evidently sample size.
    """
    matrix = frame[features].to_numpy(dtype=np.float64)
    return Reference(
        version=version,
        features=features,
        frame=frame[features],
        psi_profile=PsiProfile.fit(matrix, features, n_bins),
        score_psi=ScorePsi(frame["score"].to_numpy(dtype=np.float64), n_bins),
        evidently=EvidentlyDriftShare(frame[features], evidently_rows),
    )


class ReferenceProvider:
    """
    Tracks the champion alias and rebuilds the reference when a new champion is promoted.

    Parameters
    ----------
    registry : Registry
        Model registry.
    dataset : str
        Dataset name.
    monitored : list of str
        Feature columns the monitor receives from the flow stream.
    n_bins, evidently_rows : int
        Profile parameters.
    """

    def __init__(
        self,
        registry: Registry,
        dataset: str,
        monitored: list[str],
        n_bins: int = 10,
        evidently_rows: int = 5000,
    ):
        self.registry = registry
        self.dataset = dataset
        self.monitored = monitored
        self.n_bins = n_bins
        self.evidently_rows = evidently_rows
        self.current: Reference | None = None

    def champion(self) -> ModelRef | None:
        """Resolve the champion alias."""
        return self.registry.get_alias(model_name(self.dataset), CHAMPION)

    def refresh(self) -> bool:
        """
        Reload the reference if the champion changed.

        Returns
        -------
        changed : bool
            True when a new champion's reference was loaded (a promotion or rollback).
        """
        ref = self.champion()
        if ref is None or (self.current is not None and self.current.version == ref.version):
            return False
        with tempfile.TemporaryDirectory() as tmp:
            local = self.registry.download(ref, Path(tmp))
            frame = pd.read_parquet(local / REFERENCE_FILE)
            model_features = json.loads((local / FEATURES_FILE).read_text())
        features = [name for name in self.monitored if name in model_features]
        previous = self.current
        self.current = build_reference(
            ref.version, frame, features, self.n_bins, self.evidently_rows
        )
        return previous is not None
