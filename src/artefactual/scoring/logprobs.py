"""Logprobs processing utilities for text generation models.

This module provides functions for manipulating and processing log probabilities
from language models, including normalization, padding, and extraction.
"""

from collections.abc import Sequence
from itertools import chain
from typing import Protocol, runtime_checkable

import numpy as np
from beartype import beartype

# Type aliases - use simple types to avoid issues with plum and beartype
LogProbs = np.ndarray  # Array of log probabilities for tokens
Scores = np.ndarray  # Array of confidence scores
TokenLogProbs = Sequence[LogProbs]  # Sequence of log probability arrays (one per token)


@runtime_checkable
class LogProbValue(Protocol):
    """Protocol for objects that have a logprob attribute."""

    logprob: float


@beartype
def process_logprobs(logprobs: Sequence[Sequence[float]], max_len: int) -> LogProbs:
    """Process and pad log probabilities to a fixed length.

    This function takes sequences of log probability values and pads them
    to a consistent length for batch processing.

    Args:
        logprobs: Sequence of log probability sequences
        max_len: Maximum length to pad to

    Returns:
        Padded log probabilities as a numpy array

    Example:
        >>> import numpy as np
        >>> lps = [[0.1, 0.2], [0.3]]
        >>> process_logprobs(lps, 3)
        array([[0.1, 0.2, 0. ],
               [0.3, 0. , 0. ]], dtype=float32)
    """
    if max_len <= 0:
        msg = "max_len must be positive"
        raise ValueError(msg)

    # Handle empty input
    if not logprobs:
        return np.zeros((0, max_len), dtype=np.float32)

    # Truncate to max_len
    truncated = [np.array(lp[:max_len], dtype=np.float32) for lp in logprobs]

    # Pad to max_len
    padded = [np.pad(lp, (0, max_len - len(lp)), mode="constant") for lp in truncated]

    return np.array(padded, dtype=np.float32)


@beartype
def extract_logprobs(logprobs: Sequence[dict[int, LogProbValue]]) -> tuple[Sequence[int], Sequence[float]]:
    """Extract token IDs and log probabilities from sequence of dictionaries.

    Takes a sequence of dictionaries mapping token IDs to objects with logprob
    attributes and extracts the tokens and their corresponding log probabilities.

    Args:
        logprobs: Sequence of dictionaries mapping token IDs to objects with logprob attribute

    Returns:
        Tuple of (token IDs, log probabilities)

    Example:
        >>> class LP:
        ...     def __init__(self, p):
        ...         self.logprob = p
        >>> dicts = [{1: LP(-0.5), 2: LP(-1.2)}, {3: LP(-0.8)}]
        >>> tokens, lps = extract_logprobs(dicts)
        >>> list(tokens)
        [1, 2, 3]
        >>> list(lps)
        [-0.5, -1.2, -0.8]
    """
    if not logprobs:
        return ((), ())

    # Chain all the items from all dictionaries
    items = chain.from_iterable(lp.items() for lp in logprobs)

    # Handle empty dictionaries
    items_list = list(items)
    if not items_list:
        return ((), ())

    # Split into tokens and log probability values
    tokens, lp_values = zip(*items_list, strict=True)

    # Extract the logprob attribute from each log probability value
    lps = tuple(lp.logprob for lp in lp_values)

    return (tokens, lps)
