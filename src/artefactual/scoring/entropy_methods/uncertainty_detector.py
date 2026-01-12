from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class UncertaintyDetector(ABC):
    """A base class for uncertainty detection methods."""

    def __init__(self, k: int = 15) -> None:
        """
        Initialize the uncertainty detector.

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

    @abstractmethod
    def compute(self, inputs: Any) -> list[float]:
        """
        Compute sequence-level uncertainty scores from inputs.

        Args:
            inputs: The inputs to process (e.g. completions or model outputs).

        Returns:
            The computed sequence-level scores.
        """

    @abstractmethod
    def compute_token_scores(self, inputs: Any) -> list[NDArray[np.floating]]:
        """
        Compute token-level uncertainty scores from inputs.

        Args:
            inputs: The inputs to process (e.g. completions or model outputs).

        Returns:
            The computed token-level scores.
        """
