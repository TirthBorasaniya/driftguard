"""Custom label encoder with -1 fallback for unseen categories at inference time."""

import pickle
from pathlib import Path

import pandas as pd


class SafeLabelEncoder:
    """
    Label encoder that maps unseen categories to -1 rather than raising ValueError.

    sklearn LabelEncoder raises ValueError on unseen values and will crash the
    serving API when new categorical values appear in production. This encoder
    maps them to a reserved bin (-1) which LightGBM treats as a separate category.
    """

    def __init__(self) -> None:
        self.classes_: list[str] = []
        self._class_to_idx: dict[str, int] = {}

    def fit(self, values: pd.Series) -> "SafeLabelEncoder":
        """
        Fit encoder on a series of categorical values.

        Parameters
        ----------
        values : pd.Series
            Raw categorical values including NaN.

        Returns
        -------
        self : SafeLabelEncoder
        """
        unique_vals = sorted(
            str(v) for v in values.dropna().unique()
        )
        self.classes_ = unique_vals
        self._class_to_idx = {v: i for i, v in enumerate(unique_vals)}
        return self

    def transform(self, values: pd.Series) -> pd.Series:
        """
        Encode values. Unseen or null values map to -1.

        Parameters
        ----------
        values : pd.Series
            Values to encode.

        Returns
        -------
        encoded : pd.Series
            Integer-encoded series, dtype int32.
        """
        def _encode(v):
            if pd.isna(v):
                return -1
            return self._class_to_idx.get(str(v), -1)

        return values.map(_encode).astype("int32")

    def fit_transform(self, values: pd.Series) -> pd.Series:
        """Fit then transform in one call."""
        return self.fit(values).transform(values)

    def save(self, path: Path) -> None:
        """
        Persist encoder to disk.

        Parameters
        ----------
        path : Path
            Destination file path (pickle).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SafeLabelEncoder":
        """
        Load encoder from disk.

        Parameters
        ----------
        path : Path
            Pickle file path.

        Returns
        -------
        encoder : SafeLabelEncoder
        """
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_and_save_encoders(
    df: pd.DataFrame,
    categorical_cols: list[str],
    encoders_dir: Path,
) -> dict[str, SafeLabelEncoder]:
    """
    Fit a SafeLabelEncoder per categorical column and save to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe.
    categorical_cols : list of str
        Column names to encode.
    encoders_dir : Path
        Directory to save encoder pickles.

    Returns
    -------
    encoders_dict : dict
        Mapping of column name to fitted encoder.
    """
    encoders_dict = {}
    for col in categorical_cols:
        enc = SafeLabelEncoder()
        enc.fit(df[col])
        enc.save(encoders_dir / f"{col}_encoder.pkl")
        encoders_dict[col] = enc
        print(f"  Encoder saved: {col} ({len(enc.classes_)} classes)")
    return encoders_dict


def load_encoders(
    categorical_cols: list[str],
    encoders_dir: Path,
) -> dict[str, SafeLabelEncoder]:
    """
    Load pre-fitted encoders from disk.

    Parameters
    ----------
    categorical_cols : list of str
        Column names whose encoders to load.
    encoders_dir : Path
        Directory containing encoder pickles.

    Returns
    -------
    encoders_dict : dict
        Mapping of column name to loaded encoder.
    """
    encoders_dict = {}
    for col in categorical_cols:
        path = encoders_dir / f"{col}_encoder.pkl"
        if path.exists():
            encoders_dict[col] = SafeLabelEncoder.load(path)
        else:
            raise FileNotFoundError(f"Encoder not found: {path}")
    return encoders_dict
