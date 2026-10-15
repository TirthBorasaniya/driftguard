"""E5 gate correctness: better, equal, and worse challengers through the shadow gate (SPEC 9)."""

from typing import Any

import numpy as np
import pandas as pd

from driftguard.common.config import load_config
from driftguard.common.logging import get_logger
from driftguard.common.settings import REPO_ROOT
from driftguard.data.cicids2017 import load_dataset
from driftguard.data.registry import feature_names
from driftguard.training.gate import shadow_gate
from driftguard.training.train import predict_scores, train_model

# ============= Constants =============

logger = get_logger("benchmark.e5")


# ============= Experiment =============


def build_challengers(
    champion_rows: pd.DataFrame,
    post_train: pd.DataFrame,
    features: list[str],
    model_config: dict,
    shuffle_seed: int,
) -> dict:
    """
    Train the champion and the three synthetic challengers.

    - champion: champion days.
    - better: champion days plus the post-drift training half (newer data).
    - equal: identical data and seed to the champion.
    - worse: champion days with labels shuffled.

    Parameters
    ----------
    champion_rows : pd.DataFrame
        Rows the champion trains on.
    post_train : pd.DataFrame
        Older part of the post-drift day.
    features : list[str]
        Model feature names.
    model_config : dict
        `configs/model.yaml` content.
    shuffle_seed : int
        Seed for the label shuffle of the worse challenger.

    Returns
    -------
    models : dict
        Fitted models keyed `champion`, `better`, `equal`, and `worse`.
    """
    params, seed = model_config["lightgbm"], int(model_config["seed"])

    def fit(frame: pd.DataFrame, labels: np.ndarray | None = None):
        """Fit one model on a frame, optionally with replacement labels."""
        y = frame["label"].to_numpy() if labels is None else labels
        model, _ = train_model(frame[features].to_numpy(np.float32), y, params, seed)
        return model

    shuffled = np.random.default_rng(shuffle_seed).permutation(champion_rows["label"].to_numpy())
    return {
        "champion": fit(champion_rows),
        "better": fit(pd.concat([champion_rows, post_train], ignore_index=True)),
        "equal": fit(champion_rows),
        "worse": fit(champion_rows, shuffled),
    }


def run_e5(config_path: str) -> dict[str, Any]:
    """
    Run E5 and return its metrics.

    Parameters
    ----------
    config_path : str
        Path to `configs/eval/e5.yaml`.

    Returns
    -------
    metrics : dict
        Shadow sample sizes and the gate decision and interval for each challenger.
    """
    config = load_config(config_path).data
    model_config = load_config(REPO_ROOT / config["model_config"]).data
    rollout = load_config(REPO_ROOT / config["rollout_config"]).data
    features = feature_names(config["dataset"])
    champion_rows, _ = load_dataset(days=list(config["champion_days"]))
    post, _ = load_dataset(days=list(config["post_drift_days"]))
    # time-ordered split: the challenger trains on the older part of the post-drift day and is
    # judged on the newer part, as in live shadow traffic (a random split leaks near-duplicates)
    cut = int(len(post) * float(config["post_drift_train_fraction"]))
    post_train, pool = post.iloc[:cut], post.iloc[cut:]
    n_shadow = int(rollout["shadow_min_labeled"])
    shadow = pool.sample(n_shadow, random_state=int(config["shadow_sample_seed"]))
    models = build_challengers(
        champion_rows, post_train, features, model_config, int(config["label_shuffle_seed"])
    )
    x = shadow[features].to_numpy(np.float32)
    y = shadow["label"].to_numpy()
    champion_scores = predict_scores(models["champion"], x)
    results = {}
    for name in ("better", "equal", "worse"):
        gate = shadow_gate(
            y,
            champion_scores,
            predict_scores(models[name], x),
            float(rollout["promotion_margin"]),
            int(rollout["bootstrap_resamples"]),
            int(rollout["bootstrap_seed"]),
        )
        expected = config["expected"][name]
        results[name] = {
            **gate.as_dict(),
            "expected": expected,
            "matches_expected": gate.decision == expected,
        }
        logger.info("e5_gate", challenger=name, decision=gate.decision, expected=expected)
    return {
        "shadow_events": n_shadow,
        "shadow_positive": int(y.sum()),
        "post_drift_train_rows": len(post_train),
        "champion_rows": len(champion_rows),
        "challengers": results,
        "all_match_expected": all(r["matches_expected"] for r in results.values()),
    }
