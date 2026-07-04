"""Shadow mode: score live traffic with both champion and challenger models."""

import csv
from pathlib import Path

from src.config import FEATURE_COLS

SHADOW_MODE_MIN_EVENTS = 500  # matches DRIFT_MIN_WINDOW for consistency
SHADOW_DIVERGENCE_THRESHOLD = 0.05  # max acceptable prediction disagreement rate


def score_shadow_mode(
    champion_model,
    challenger_model,
    feature_dict: dict[str, float],
) -> tuple[float, float]:
    """
    Score a single event with both models without exposing the challenger
    prediction to the serving response.

    Parameters
    ----------
    champion_model : LGBMClassifier
        Currently promoted model, used for the actual served prediction.
    challenger_model : LGBMClassifier
        Candidate model under shadow evaluation.
    feature_dict : dict[str, float]
        Computed feature values for the event.

    Returns
    -------
    champion_pred : float
        Champion's predicted probability, this is what gets served.
    challenger_pred : float
        Challenger's predicted probability, logged only, not served.
    """
    feature_values = [[feature_dict[col] for col in FEATURE_COLS]]

    champion_pred = float(champion_model.predict_proba(feature_values)[0, 1])
    challenger_pred = float(challenger_model.predict_proba(feature_values)[0, 1])

    return champion_pred, challenger_pred


def log_shadow_prediction(
    shadow_log_path: str,
    champion_pred: float,
    challenger_pred: float,
    champion_decision: bool,
    challenger_decision: bool,
) -> None:
    """
    Append a champion/challenger prediction pair to the shadow log.

    Parameters
    ----------
    shadow_log_path : str
        Path to the CSV file logging shadow prediction pairs.
    champion_pred : float
        Champion's predicted probability.
    challenger_pred : float
        Challenger's predicted probability.
    champion_decision : bool
        Champion's binary decision at its calibrated threshold.
    challenger_decision : bool
        Challenger's binary decision at its calibrated threshold.
    """
    path = Path(shadow_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    o_write_header = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if o_write_header:
            writer.writerow(
                ["champion_pred", "challenger_pred", "champion_decision", "challenger_decision"]
            )
        writer.writerow(
            [champion_pred, challenger_pred, int(champion_decision), int(challenger_decision)]
        )


def evaluate_shadow_divergence(
    shadow_log_path: str,
    min_events: int,
    divergence_threshold: float,
) -> bool:
    """
    Compute prediction disagreement rate between champion and challenger over
    the shadow log and decide whether promotion should proceed.

    Parameters
    ----------
    shadow_log_path : str
        Path to the logged champion/challenger prediction pairs.
    min_events : int
        Minimum shadow events required before evaluating.
    divergence_threshold : float
        Maximum acceptable disagreement rate for promotion to proceed.

    Returns
    -------
    o_promotion_approved : bool
        True if divergence is within threshold and enough events were logged.
    """
    path = Path(shadow_log_path)
    if not path.exists():
        return False

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < min_events:
        return False

    disagreements = sum(
        1 for row in rows if row["champion_decision"] != row["challenger_decision"]
    )
    divergence_rate = disagreements / len(rows)

    o_promotion_approved = divergence_rate <= divergence_threshold
    return o_promotion_approved
