import numpy as np
from beartype import beartype


class EntropyContributionsMixin:
    """
    Mixin that provides entropy contribution computation for top-K logprobs.
    """

    @staticmethod
    @beartype
    def entropy_contributions(logprobs: np.ndarray) -> np.ndarray:
        """Compute entropic contributions s_kj = -p_k log(p_k) for top-K logprobs using vectorized operations.
        Args:
            logprobs: An array of log probabilities, with the last axis being the per-token rank axis.

        Returns:
            An array of the same shape containing entropy contributions.
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


@beartype
def align_rank_width(contributions: np.ndarray, k: int) -> np.ndarray:
    """Truncate or zero-pad entropy contributions to exactly `k` ranks.

    Calibrated weights are fixed at the rank count used during calibration, so the rank
    axis must match before the weighted sum. Zero is the limit of the contribution itself
    -- -p*log(p) tends to 0 as p tends to 0 -- so absent ranks add nothing to the score.

    Only the trailing axis is touched, so this serves both the (num_tokens, num_ranks)
    shape the scorers build per sequence and the (n_sequences, num_tokens, num_ranks)
    cube the pipeline passes between steps.

    Args:
        contributions: Array whose last axis is the rank axis.
        k: Number of ranks the calibrated weights expect.

    Returns:
        Array with the same leading axes and exactly `k` ranks.

    Raises:
        ValueError: If `k` is not positive, or the array has no rank axis.
    """
    if k <= 0:
        msg = f"k must be positive, got {k}"
        raise ValueError(msg)
    if contributions.ndim < 1:
        msg = "contributions must have at least one axis (the rank axis)"
        raise ValueError(msg)

    num_ranks = contributions.shape[-1]
    if num_ranks == k:
        return contributions
    if num_ranks > k:
        return contributions[..., :k]

    aligned = np.zeros((*contributions.shape[:-1], k), dtype=contributions.dtype)
    aligned[..., :num_ranks] = contributions
    return aligned
