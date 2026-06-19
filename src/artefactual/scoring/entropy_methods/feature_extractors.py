from typing import NoReturn

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from artefactual.scoring.entropy_methods.entropy_contributions import EntropyContributionsMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 15) -> None:
        self.k = k

    def fit(self, _x, _y=None) -> "EPRFeatureExtractor":
        return self

    @property
    def fallback_value(self) -> NoReturn:
        """
        Subclasses must implement what to return if a sample is empty.
        """
        raise NotImplementedError

    def transform(self, x: list[dict[int, list[float]]]) -> np.ndarray:
        """
        Transforms the input data into a feature matrix.

        Returns:
        np.ndarray: A feature matrix of shape (n_samples, 2 * k).
        """
        features = []
        for token_logprobs in x:
            if not token_logprobs:
                features.append(self.fallback_value)
                continue
            sorted_indices = sorted(token_logprobs.keys())
            logprobs_list = [token_logprobs[i] for i in sorted_indices]
            s_kj = EntropyContributionsMixin.compute_entropy_contributions(logprobs_list, self.k)  # (n_tokens, k)
            features.append(self.reduce(s_kj))  # (2k,)
        return np.vstack(features, dtype=np.float32)  # (n_samples, 2k)

    def _more_tags(self) -> dict[str, bool]:
        return {"no_validation": True}


class EPRFeatureExtractor(FeatureExtractor):
    @property
    def fallback_value(self) -> float:
        return 0.0  # Scalar fallback

    def reduce(self, entropy_contributions) -> np.float32:
        return np.mean(np.sum(entropy_contributions, axis=1))


class WEPRFeatureExtractor(FeatureExtractor):
    """
    WEPR feature extractor. Produces a feature matrix of shape (n_samples, 2 * k).
    """

    @property
    def fallback_value(self) -> np.ndarray:
        return np.zeros(2 * self.k, dtype=np.float32)

    def reduce(self, entropy_contributions) -> np.ndarray:
        return np.concatenate([np.mean(entropy_contributions, axis=0), np.max(entropy_contributions, axis=0)])
