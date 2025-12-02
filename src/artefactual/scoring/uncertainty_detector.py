from typing import Protocol, runtime_checkable


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
