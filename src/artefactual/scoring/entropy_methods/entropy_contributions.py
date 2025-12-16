from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

EPSILON = 1e-12


@beartype
def compute_entropy_contributions(logprobs: NDArray[np.floating] | Sequence[Any], k: int) -> NDArray[np.floating]:
    """Compute entropic contributions s_kj = -p_k log(p_k) for top-K logprobs using vectorized operations.
    Args:
        logprobs: A 2D array of shape (num_tokens, num_logprobs) containing log probabilities.
        k: Number of top log probabilities to consider per token.

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
                    vals = list(row.values()) if isinstance(row, Mapping) else row
                    # Handle objects with logprob attribute (e.g. vLLM Logprob objects)
                    vals = [v.logprob if hasattr(v, "logprob") else v for v in vals]
                    padded_logprobs[i, : len(vals)] = vals
            logprobs = padded_logprobs

    if logprobs.size == 0:
        return np.empty((0, k), dtype=np.float32)

    # Convert to probabilities (logprobs are in natural log, base e)
    probs = np.exp(logprobs)

    # Calculate entropy contributions in nats (use log_e)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -probs * logprobs
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad or truncate to k elements along the K dimension (axis=1)
    num_tokens, num_logprobs = s.shape
    if num_logprobs == k:
        return s

    s_kj = np.zeros((num_tokens, k), dtype=np.float32)
    if num_logprobs < k:
        s_kj[:, :num_logprobs] = s
    else:  # num_logprobs > k
        s_kj[:, :] = s[:, :k]

    return s_kj
