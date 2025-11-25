from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

EPSILON = 1e-12


@runtime_checkable
class LogProb(Protocol):
    """A protocol for an object with a log probability."""

    logprob: float


class UncertaintyDetector:
    """A base class for uncertainty detection methods."""

    def __init__(self, k: int = 15) -> None:
        """Initialize the uncertainty detector.
        Args:
            k: Number of top log probabilities to consider per token.
               Must be positive. Default is 15.
        Raises:
            ValueError: If k is not positive
        """
        if k <= 0:
            msg = f"k must be positive, got {k}"
            raise ValueError(msg)
        self.k = k

    @beartype
    def _entropy_contributions(
        self, logprobs: NDArray[np.floating] | Sequence[Sequence[float]]
    ) -> NDArray[np.floating]:
        """Compute entropic contributions s_kj = -p_k log2(p_k) for top-K logprobs using vectorized operations.
        Args:
            logprobs: A 2D array of shape (num_tokens, num_logprobs) containing log probabilities.
        Returns:
            A 2D array of shape (num_tokens, K) containing entropy contributions.
        """
        if not isinstance(logprobs, np.ndarray):
            if not logprobs:
                logprobs = np.empty((0, 0), dtype=np.float32)
            else:
                # Handle potential ragged sequences by padding with -inf
                max_len = max(len(row) for row in logprobs)
                padded_logprobs = np.full((len(logprobs), max_len), -np.inf, dtype=np.float32)
                for i, row in enumerate(logprobs):
                    if row:
                        padded_logprobs[i, : len(row)] = row
                logprobs = padded_logprobs

        if logprobs.size == 0:
            return np.empty((0, self.k), dtype=np.float32)

        # Convert to probabilities (logprobs are in natural log, base e)
        probs = np.exp(logprobs)

        # Normalize top-K probs to sum to 1 along the K dimension (axis=1)
        probs_sum = probs.sum(axis=1, keepdims=True)
        # Avoid division by zero for tokens with no logprobs
        probs_sum[probs_sum == 0] = 1.0
        probs /= probs_sum

        # Calculate entropy contributions in bits (use log2)
        # s = -p * log2(p), with special handling for p=0
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -probs * np.log2(probs + EPSILON)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad or truncate to k elements along the K dimension (axis=1)
        num_tokens, num_logprobs = s.shape
        if num_logprobs == self.k:
            return s

        s_kj = np.zeros((num_tokens, self.k), dtype=np.float32)
        if num_logprobs < self.k:
            s_kj[:, :num_logprobs] = s
        else:  # num_logprobs > self.k
            s_kj[:, :] = s[:, : self.k]

        return s_kj
