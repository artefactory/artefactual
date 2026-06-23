import numpy as np
from beartype import beartype
from numpy.typing import NDArray

EPSILON = 1e-12


class EntropyContributionsMixin:
    """
    Mixin that provides entropy contribution computation for top-K logprobs.
    """

    @staticmethod
    @beartype
    def entropy_contributions(logprobs: NDArray[np.floating], k: int) -> NDArray[np.floating]:
        """Compute entropic contributions s_kj = -p_k log(p_k) for top-K logprobs using vectorized operations.
        Args:
            logprobs: A 2D array of shape (num_tokens, num_logprobs) containing log probabilities.
            k: Number of top log probabilities to consider per token.

        Returns:
            A 2D array of shape (num_tokens, K) containing entropy contributions.
        """

        if logprobs.size == 0:
            return np.empty((0, k), dtype=np.float32)

        # Convert to probabilities (logprobs are in natural log, base e)
        probs = np.exp(logprobs)

        # Calculate entropy contributions: s = -p * log(p) = -exp(logp) * logp (logprobs are natural logs)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -probs * logprobs
