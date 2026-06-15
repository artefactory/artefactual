import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from artefactual.scoring.entropy_methods.entropy_contributions import compute_entropy_contributions


class EPRFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    EPR feature extractor. Produces a feature matrix of shape (n_samples, 1).

    For each sequence: sums entropy contributions across rank K per token to get
    a per-token EPR score, then takes the mean across all tokens.

    Accepts list[dict[int, list[float]]].
    Bypasses sklearn array validation via _more_tags to handle this non-standard input.
    """

    def __init__(self, k: int = 15) -> None:
        self.k = k

    def fit(self, _x, _y=None) -> "EPRFeatureExtractor":
        return self

    def transform(self, x: list[dict[int, list[float]]]) -> np.ndarray:
        features = []
        for token_logprobs in x:
            if not token_logprobs:
                features.append(0.0)
                continue
            sorted_indices = sorted(token_logprobs.keys())
            logprobs_list = [token_logprobs[i] for i in sorted_indices]
            s_kj = compute_entropy_contributions(logprobs_list, self.k)  # (n_tokens, k)
            features.append(np.mean(np.sum(s_kj, axis=1)))
        return np.array(features, dtype=np.float32).reshape(-1, 1)  # (n_samples, 1)

    def _more_tags(self) -> dict[str, bool]:
        return {"no_validation": True}


class WEPRFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    WEPR feature extractor. Produces a feature matrix of shape (n_samples, 2 * k).
    """

    def __init__(self, k: int = 15) -> None:
        self.k = k

    def fit(self, _x, _y=None) -> "WEPRFeatureExtractor":
        return self

    def transform(self, x: list[dict[int, list[float]]]) -> np.ndarray:
        """
        Transforms the input data into a feature matrix.

        Returns:
        np.ndarray: A feature matrix of shape (n_samples, 2 * k).
        """
        features = []
        for token_logprobs in x:
            if not token_logprobs:
                features.append(np.zeros(2 * self.k, dtype=np.float32))
                continue
            sorted_indices = sorted(token_logprobs.keys())
            logprobs_list = [token_logprobs[i] for i in sorted_indices]
            s_kj = compute_entropy_contributions(logprobs_list, self.k)  # (n_tokens, k)
            features.append(np.concatenate([np.mean(s_kj, axis=0), np.max(s_kj, axis=0)]))  # (2k,)
        return np.array(features, dtype=np.float32)  # (n_samples, 2k)

    def _more_tags(self) -> dict[str, bool]:
        """
        Configures scikit-learn estimator tags to bypass default array conversion validation.
        """
        return {"no_validation": True}
