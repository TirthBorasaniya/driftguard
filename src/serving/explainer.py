"""SHAP TreeExplainer wrapper returning top-N feature contributions per prediction."""

import numpy as np


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer for per-prediction feature attribution.

    Used by the /predict/explain endpoint to satisfy explainability
    requirements common in production fraud detection deployments.
    """

    def __init__(self, model) -> None:
        import shap

        self.explainer = shap.TreeExplainer(model)

    def top_features(
        self,
        X: np.ndarray,
        feature_names: list[str],
        n: int = 5,
    ) -> list[dict]:
        """
        Compute top-N SHAP contributions for a single prediction.

        Parameters
        ----------
        X : np.ndarray
            Feature vector, shape (1, n_features).
        feature_names : list of str
            Feature names in the same order as X columns.
        n : int
            Number of top contributions to return.

        Returns
        -------
        contributions : list of dict
            Sorted by abs(shap_value) descending.
            Each dict has keys: feature, shap_value.
        """
        shap_values = self.explainer.shap_values(X)

        # shap_values may be list (binary) or array — take positive class
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]

        n_features = min(len(feature_names), len(values))
        contributions = [
            {"feature": feature_names[i], "shap_value": float(values[i])}
            for i in range(n_features)
        ]
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions[:n]
