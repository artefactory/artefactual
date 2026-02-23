import numpy as np
from numpy.typing import NDArray

from artefactual.scoring.uncertainty_detector import SentenceProbabilityDetector


class SentenceProbabilityScorer(SentenceProbabilityDetector):
    """
    Computes sentence-level probability from the sampled tokens log probabilities.
    This method aggregates token-level log probabilities into a single score for the entire sequence.
    You can parse raw model outputs using the `parse_sampled_token_logprobs` method from `artefactual.preprocessing`.
    """

    def compute(self, inputs: NDArray[np.floating]) -> list[float]:
        """
        Compute sentence-level probability scores by summing token log probabilities.

        Args:
            inputs: A list of token log probabilities for each token in the sequence.
        Returns:
            The whole sentence probability.
        """
        sentence_scores = [np.sum(seq) for seq in inputs]
        return np.exp(sentence_scores).tolist()  # Convert log probability to probability

    def compute_token_scores(self, inputs: NDArray[np.floating]) -> list[NDArray[np.floating]]:
        """
        Compute sentence-level probability scores by summing token log probabilities.

        Args:
            inputs: A list of token log probabilities for each token in the sequence.

        Returns:
            A list of numpy arrays of token-level probabilities.
        """
        return [np.exp(seq) for seq in inputs]
