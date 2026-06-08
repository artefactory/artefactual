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
        means = []
        for ts in token_scores:
            if ts.size > 0:
                means.append(np.mean(ts))
            else:
                means.append(0.0)
        return np.array(means, dtype=np.float32).reshape(-1, 1)  # (n_samples, 1)

    def compute_token_scores(self, x: list[dict[int, list[float]]]) -> list[NDArray]:
        """
        extension method, not part of the sklearn contract.
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
