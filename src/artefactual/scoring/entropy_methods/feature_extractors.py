import numpy as np
from numpy.typing import NDArray
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
        token_scores = self.compute_token_scores(x)
        features = []
        for ts in token_scores:
            if ts.size > 0:
                features.append(np.mean(ts))
            else:
                features.append(0.0)
        return np.array(features, dtype=np.float32).reshape(-1, 1)  # (n_samples, 1)

    def compute_token_scores(self, x: list[dict[int, list[float]]]) -> list[NDArray]:
        """
        Extension method, not part of the sklearn contract.
        Output is a ragged list of arrays.
        """
        raw_token_scores = []
        for token_logprobs in x:
            if not token_logprobs:
                raw_token_scores.append(np.array([], dtype=np.float32))
                continue
            sorted_indices = sorted(token_logprobs.keys())
            logprobs_list = [token_logprobs[i] for i in sorted_indices]
            s_kj = compute_entropy_contributions(logprobs_list, self.k)  # (n_tokens, k)
            raw_token_scores.append(np.sum(s_kj, axis=1))
        return raw_token_scores

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
        for s_kj in self.compute_token_scores(x):
            if len(s_kj) == 0:
                features.append(np.zeros(2 * self.k, dtype=np.float32))
            else:
                features.append(np.concatenate([np.mean(s_kj, axis=0), np.max(s_kj, axis=0)]))  # (2k,)
        return np.array(features, dtype=np.float32)  # (n_samples, 2k)

    def compute_token_scores(self, x: list[dict[int, list[float]]]) -> list[NDArray]:
        """
        Computes the raw entropic contributions (s_kj) for each token candidate across the sequence.

        This is an extension method (not part of the sklearn contract) used to extract granular, token-level
        metrics before any sequence-level averaging or weighting is applied.
        """
        raw_token_scores = []
        for token_logprobs in x:
            if not token_logprobs:
                raw_token_scores.append(np.array([], dtype=np.float32))
                continue
            sorted_indices = sorted(token_logprobs.keys())
            logprobs_list = [token_logprobs[i] for i in sorted_indices]
            s_kj = compute_entropy_contributions(logprobs_list, self.k)  # (n_tokens, k)
            raw_token_scores.append(s_kj)
        return raw_token_scores

    def _more_tags(self) -> dict[str, bool]:
        """
        Configures scikit-learn estimator tags to bypass default array conversion validation.
        """
        return {"no_validation": True}
