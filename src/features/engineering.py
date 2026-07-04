"""Single source of truth for network flow feature transformations at training and serving time."""

import pandas as pd

# ============= Feature Definitions =============

# canonical model feature list; imported by config, drift_detector, training, and serving
# so the same ten columns are computed identically across every path
FEATURE_COLS = [
    "flow_duration",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "total_fwd_packets",
    "total_bwd_packets",
    "packet_length_mean",
    "packet_length_std",
    "flow_iat_mean",
    "fwd_bwd_packet_ratio",  # engineered
    "syn_flag_count",
]

# pass-through columns taken directly from the event with no computation
PASSTHROUGH_COLS = [
    "flow_duration",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "total_fwd_packets",
    "total_bwd_packets",
    "packet_length_mean",
    "packet_length_std",
    "flow_iat_mean",
    "syn_flag_count",
]

EPSILON = 1e-9  # division guard for ratio features


# ============= Single Event (serving path) =============


def compute_features(event: dict) -> dict[str, float]:
    """
    Compute all ten features from a single network flow event dict.

    Used by the Kafka consumer and the FastAPI serving layer so that the
    feature vector built at inference time matches the training feature
    matrix exactly, preventing training-serving skew.

    Parameters
    ----------
    event : dict
        Raw network flow event conforming to the NetworkFlowEvent schema.

    Returns
    -------
    feature_dict : dict[str, float]
        Mapping of feature name to computed float value for all FEATURE_COLS.
    """
    # pass-through fields default to 0.0 when absent so an incomplete event
    # degrades gracefully rather than raising at inference time
    feature_dict = {col: float(event.get(col, 0.0) or 0.0) for col in PASSTHROUGH_COLS}

    fwd = float(event.get("total_fwd_packets", 0.0) or 0.0)
    bwd = float(event.get("total_bwd_packets", 0.0) or 0.0)
    feature_dict["fwd_bwd_packet_ratio"] = fwd / (bwd + EPSILON)

    # return in canonical FEATURE_COLS order
    return {col: feature_dict[col] for col in FEATURE_COLS}


# ============= Batch (training path) =============


def compute_features_batch(event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all ten features for a batch DataFrame of network flow events.

    Parameters
    ----------
    event_df : pd.DataFrame
        DataFrame of raw network flow events with NetworkFlowEvent columns.

    Returns
    -------
    feature_df : pd.DataFrame
        DataFrame with FEATURE_COLS columns, indexed identically to event_df.
    """
    feature_df = pd.DataFrame(index=event_df.index)

    for col in PASSTHROUGH_COLS:
        # coerce to float and fill missing with 0.0 to mirror compute_features
        feature_df[col] = pd.to_numeric(event_df.get(col), errors="coerce").fillna(0.0).astype("float64")

    fwd = pd.to_numeric(event_df.get("total_fwd_packets"), errors="coerce").fillna(0.0)
    bwd = pd.to_numeric(event_df.get("total_bwd_packets"), errors="coerce").fillna(0.0)
    feature_df["fwd_bwd_packet_ratio"] = (fwd / (bwd + EPSILON)).astype("float64")

    return feature_df[FEATURE_COLS]
