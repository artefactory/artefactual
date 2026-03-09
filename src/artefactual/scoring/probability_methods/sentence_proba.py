import numpy as np
from numpy.typing import NDArray

from artefactual.scoring.uncertainty_detector import SentenceProbabilityDetector


class SentenceProbabilityScorer(SentenceProbabilityDetector):
    """
    Computes sentence-level probability from the sampled tokens log probabilities.
    This method aggregates token-level log probabilities into a single score for the entire sequence.
    You can parse raw model outputs using the `parse_sampled_token_logprobs` method from `artefactual.preprocessing`.
    """

    def compute(self, inputs: list[NDArray[np.floating]]) -> list[float]:
        """
        Compute sentence-level probability scores by summing token log probabilities.
        Empty sequences are treated as out-of-domain inputs and mapped to a 0.0 fallback.

        Args:
            inputs: A list of 1D numpy arrays of token log probabilities, one per sequence.
        Returns:
            The whole sentence probability for each sequence.
        """

        return [0.0 if len(seq) == 0 else float(np.exp(np.sum(seq))) for seq in inputs]

    def compute_token_scores(self, inputs: list[NDArray[np.floating]]) -> list[NDArray[np.floating]]:
        """
        Returns the sampled token probabilities (exponentiating token logprobs).

        Args:
            inputs: A list of token log probabilities for each token in the sequence.

        Returns:
            A list of numpy arrays of token-level probabilities.
        """
        return [np.exp(seq) for seq in inputs]
