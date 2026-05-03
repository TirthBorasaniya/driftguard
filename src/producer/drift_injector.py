"""Drift injection: modify transaction records to simulate distributional shift."""

import json
import random
from pathlib import Path


def load_scenario(scenario_path: Path) -> dict:
    """
    Load a drift scenario config from JSON.

    Parameters
    ----------
    scenario_path : Path
        Path to scenario JSON file.

    Returns
    -------
    scenario : dict
    """
    with open(scenario_path) as f:
        return json.load(f)


def inject_drift(record: dict, scenario: dict) -> dict:
    """
    Apply drift transformations to a transaction record.

    Modifies amount, category distribution, and state concentration
    according to the scenario config.

    Parameters
    ----------
    record : dict
        Original transaction record.
    scenario : dict
        Drift scenario config loaded from JSON.

    Returns
    -------
    record : dict
        Modified record with injected drift.
    """
    record = dict(record)

    # scale transaction amount
    amt_multiplier = scenario.get("amt_multiplier", 1.0)
    if "amt" in record:
        record["amt"] = float(record["amt"]) * amt_multiplier

    # shift category distribution
    category_shift = scenario.get("category_shift", {})
    if category_shift:
        roll = random.random()
        cumulative = 0.0
        for cat, prob in category_shift.items():
            cumulative += prob
            if roll < cumulative:
                record["category"] = cat
                break

    # concentrate transactions in one state
    state_concentration = scenario.get("state_concentration")
    state_rate = scenario.get("state_concentration_rate", 0.7)
    if state_concentration and random.random() < state_rate:
        record["state"] = state_concentration

    return record
