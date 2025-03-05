"""Logprobs processing utilities."""

from collections.abc import Sequence
from itertools import chain
from typing import Protocol, TypeVar

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

# Type aliases
Logprobs = NDArray[np.float32]
Scores = NDArray[np.float32]


class LogProbValue(Protocol):
    """Protocol for objects that have a logprob attribute."""

    logprob: float


T = TypeVar("T", bound=LogProbValue)


@beartype
def process_logprobs(logprobs: Sequence[Sequence[float]], max_len: int) -> Logprobs:
    """Process and pad log probabilities to a fixed length.

    Args:
        logprobs: Sequence of log probability sequences
        max_len: Maximum length to pad to

    Returns:
        Padded log probabilities as a numpy array
    """
    logprobs = (lp[:max_len] for lp in logprobs)
    logprobs = [np.pad(lp, (0, max_len - len(lp)), mode="constant") for lp in logprobs]
    return np.array(logprobs, dtype=np.float32)


@beartype
def extract_logprobs(logprobs: Sequence[dict[int, LogProbValue]]) -> tuple[Sequence[int], Sequence[float]]:
    """Extract token IDs and log probabilities from sequence of dictionaries.

    Args:
        logprobs: Sequence of dictionaries mapping token IDs to objects with logprob attribute

    Returns:
        Tuple of (token IDs, log probabilities)
    """
    tokens, lps = zip(*chain.from_iterable([lp.items() for lp in logprobs]), strict=True)
    return (tokens, tuple(lp.logprob for lp in lps))
