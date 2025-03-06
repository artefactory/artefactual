"""Entropy calculation functions for statistical analysis of model outputs.

This module provides functions for calculating entropy-based metrics from
log probability distributions, useful for analyzing model uncertainty and confidence.
"""

from collections.abc import Sequence

import numpy as np
from beartype import beartype
from numpy.typing import ArrayLike, NDArray

# More flexible type for LogProbs (allow both float32 and float64)
LogProbsInput = Sequence[float] | NDArray[np.floating]


@beartype
def entropy_from_logprobs(logprobs: LogProbsInput) -> float:
    """Calculate entropy from log probabilities.

    Entropy is a measure of uncertainty in a probability distribution.
    Higher entropy indicates higher uncertainty.

    Args:
        logprobs: Log probabilities from model output

    Returns:
        Calculated entropy value (non-negative float)

    Raises:
        ValueError: If the input is empty
        ValueError: If the input contains inf or nan values

    Example:
        >>> import numpy as np
        >>> logprobs = np.array([-0.7, -1.2, -2.3])
        >>> entropy_from_logprobs(logprobs)
        0.94... # approximate value
    """
    # Convert to numpy array if not already
    logprobs_array = np.asarray(logprobs, dtype=np.float32)

    if logprobs_array.size == 0:
        msg = "Cannot calculate entropy from empty log probabilities"
        raise ValueError(msg)

    # Check for inf or nan values
    if np.any(~np.isfinite(logprobs_array)):
        msg = "Log probabilities cannot contain inf or nan values"
        raise ValueError(msg)

    # Handle numerical stability with np.exp for log probabilities
    probs = np.exp(logprobs_array)
    return float(-np.sum(probs * logprobs_array))


@beartype
def token_entropy(
    token_logprobs: Sequence[LogProbsInput],
) -> list[float]:
    """Calculate entropy for each token position.

    Args:
        token_logprobs: Sequence of token-level log probabilities

    Returns:
        List of entropy values for each token

    Raises:
        ValueError: If any token's log probabilities are empty

    Example:
        >>> import numpy as np
        >>> token_logprobs = [np.array([-0.7, -1.2, -2.3]), np.array([-0.3, -1.5])]
        >>> token_entropy(token_logprobs)
        [0.94..., 0.65...] # approximate values
    """
    if not token_logprobs:
        return []

    return [entropy_from_logprobs(lp) for lp in token_logprobs]


@beartype
def mean_entropy(token_logprobs: Sequence[LogProbsInput], weights: ArrayLike | None = None) -> float:
    """Calculate the mean entropy across all tokens.

    Args:
        token_logprobs: Sequence of token-level log probabilities
        weights: Optional weights for each token position

    Returns:
        Mean entropy value (or weighted mean if weights are provided)

    Raises:
        ValueError: If token_logprobs is empty
        ValueError: If weights length doesn't match token_logprobs length

    Example:
        >>> import numpy as np
        >>> token_logprobs = [np.array([-0.7, -1.2, -2.3]), np.array([-0.3, -1.5])]
        >>> mean_entropy(token_logprobs)
        0.80... # approximate value

        >>> weights = [0.7, 0.3]
        >>> mean_entropy(token_logprobs, weights)
        0.85... # approximate value
    """
    if not token_logprobs:
        msg = "Cannot calculate mean entropy from empty sequence"
        raise ValueError(msg)

    entropies = token_entropy(token_logprobs)

    if weights is not None:
        weights_array = np.asarray(weights, dtype=np.float32)
        if len(weights_array) != len(entropies):
            error_msg = f"Weights length {len(weights_array)} must match entropies length {len(entropies)}"
            raise ValueError(error_msg)
        return float(np.average(entropies, weights=weights_array))

    return float(sum(entropies) / len(entropies))
