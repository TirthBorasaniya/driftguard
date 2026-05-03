"""Single source of truth for all feature transformations applied at training and serving time."""

import numpy as np
import pandas as pd


# ============= Derived Feature Transforms =============


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorized haversine distance in kilometres between two coordinate pairs.

    Parameters
    ----------
    lat1, lon1 : np.ndarray
        Cardholder coordinates in decimal degrees.
    lat2, lon2 : np.ndarray
        Merchant coordinates in decimal degrees.

    Returns
    -------
    distance : np.ndarray
        Great-circle distance in kilometres.
    """
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_age(
    dob_series: pd.Series,
    reference_timestamps: pd.Series,
) -> pd.Series:
    """
    Compute cardholder age in years at transaction time.

    Parameters
    ----------
    dob_series : pd.Series
        Date of birth strings (YYYY-MM-DD format).
    reference_timestamps : pd.Series
        Transaction timestamps.

    Returns
    -------
    age : pd.Series
        Age in whole years, dtype float32.
    """
    dob = pd.to_datetime(dob_series, errors="coerce")
    ts = pd.to_datetime(reference_timestamps, errors="coerce")
    age = (ts - dob).dt.days / 365.25
    return age.fillna(-999).astype("float32")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all derived feature transformations.

    Called at both training time (preprocess.py) and serving time (kafka_consumer.py).
    Any change here applies to both paths, preventing training-serving skew.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataframe with Sparkov schema columns.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with derived columns added in-place (copy returned).
    """
    df = df.copy()

    ts = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")

    df["hour_of_day"] = ts.dt.hour.astype("float32")
    df["day_of_week"] = ts.dt.dayofweek.astype("float32")

    df["age"] = compute_age(df["dob"], df["trans_date_trans_time"])

    df["distance_km"] = haversine_km(
        df["lat"].values.astype(float),
        df["long"].values.astype(float),
        df["merch_lat"].values.astype(float),
        df["merch_long"].values.astype(float),
    ).astype("float32")

    df["amt_log"] = np.log1p(df["amt"].clip(lower=0)).astype("float32")

    return df


def engineer_single_event(event: dict) -> dict:
    """
    Apply derived feature transforms to a single transaction dict.

    Used in the Kafka consumer for per-event serving-time feature computation.

    Parameters
    ----------
    event : dict
        Single transaction with raw Sparkov fields.

    Returns
    -------
    event : dict
        Event dict with derived fields added.
    """
    import math
    from datetime import datetime

    event = dict(event)

    try:
        ts = pd.to_datetime(event["trans_date_trans_time"])
        event["hour_of_day"] = float(ts.hour)
        event["day_of_week"] = float(ts.dayofweek)
    except Exception:
        event["hour_of_day"] = -999.0
        event["day_of_week"] = -999.0

    try:
        dob = pd.to_datetime(event["dob"])
        ts = pd.to_datetime(event["trans_date_trans_time"])
        event["age"] = float((ts - dob).days / 365.25)
    except Exception:
        event["age"] = -999.0

    try:
        lat1 = float(event["lat"])
        lon1 = float(event["long"])
        lat2 = float(event["merch_lat"])
        lon2 = float(event["merch_long"])
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        )
        event["distance_km"] = 2 * R * math.asin(math.sqrt(max(0, min(1, a))))
    except Exception:
        event["distance_km"] = -999.0

    try:
        event["amt_log"] = float(math.log1p(max(0, float(event["amt"]))))
    except Exception:
        event["amt_log"] = -999.0

    return event
