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
    def entropy_contributions(logprobs: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute entropic contributions s_kj = -p_k log(p_k) for top-K logprobs using vectorized operations.
        Args:
            logprobs: An array of log probabilities, the last axis being the per-token rank axis.

        Returns:
            An array of the same shape as `logprobs` containing entropy contributions.
        """

        if logprobs.size == 0:
            return np.empty_like(logprobs)

        # Enforce descending rank order along the rank axis.
        logprobs = -np.sort(-logprobs, axis=-1)

        # Convert to probabilities (logprobs are in natural log, base e)
        probs = np.exp(logprobs)

        # Calculate entropy contributions: s = -p * log(p) = -exp(logp) * logp (logprobs are natural logs)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -probs * logprobs
