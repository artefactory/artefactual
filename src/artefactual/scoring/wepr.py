from collections.abc import Sequence

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from artefactual.data.data_model import Completion
from artefactual.features.entropy_contributions import compute_entropy_contributions
from artefactual.scoring.uncertainty_detector import UncertaintyDetector
from artefactual.utils.io import load_weights


class WEPR(UncertaintyDetector):
    def __init__(self, model: str) -> None:
        """
        Initialize the WEPR scorer with weights loaded from the specified source.
        Args:
            model: Either a built-in model name or a local file path to load weights from.
        """
        weights_data = load_weights(model)
        self.intercept = weights_data.get("intercept", 0.0)
        coeffs = weights_data.get("coefficients", {})

        # Determine k from the coefficients (assuming keys like "mean_rank_15")
        # We look for the maximum rank index present in the coefficients
        ranks = [
            int(key.split("_")[-1])
            for key in coeffs.keys()
            if key.startswith("mean_rank_") and key.split("_")[-1].isdigit()
        ]
        k = max(ranks) if ranks else 15

        super().__init__(k=k)

        # Parse weights into numpy arrays for vectorized computation
        self.mean_weights = np.zeros(k, dtype=np.float32)
        self.max_weights = np.zeros(k, dtype=np.float32)

        for i in range(1, k + 1):
            self.mean_weights[i - 1] = coeffs.get(f"mean_rank_{i}", 0.0)
            self.max_weights[i - 1] = coeffs.get(f"max_rank_{i}", 0.0)

    @beartype
    def compute(
        self,
        completions: Sequence[Completion],
        *,
        return_per_token_scores: bool = False,
    ) -> list[float] | tuple[list[float], list[NDArray[np.floating]]]:
        """
        Compute WEPR-based uncertainty scores from a sequence of completions.
        Args:
            completions: A list of completions, where each completion is a list of tokens,
                         and each token is a list of its top-K log probabilities.
                         Shape: (num_completions, num_tokens, num_logprobs)
        Returns:
            - List of sequence-level WEPR scores.
            - Optionally (if return_per_token_scores=True),
              a tuple of (seq_scores, per_token_scores).
        """
        if not completions:
            return []

        seq_scores: list[float] = []
        token_scores: list[NDArray[np.floating]] = []

        for completion in completions:
            token_logprobs_dict = completion.token_logprobs
            if not token_logprobs_dict:
                # If no tokens, return the intercept (or 0.0)
                seq_scores.append(self.intercept)
                if return_per_token_scores:
                    token_scores.append(np.array([], dtype=np.float32))
                continue

            # Convert to a 2D numpy array for vectorized processing
            # Sort by token position to ensure correct order
            sorted_indices = sorted(token_logprobs_dict.keys())
            logprobs_list = [token_logprobs_dict[i] for i in sorted_indices]

            # Compute entropy contributions in a vectorized manner
            # Input shape: (num_tokens, K)
            s_kj = compute_entropy_contributions(logprobs_list, self.k)

            # Token-level WEPR (S_beta): weighted sum across K using mean_weights + intercept
            # Eq (7): S_beta = beta_0 + sum(beta_k * s_kj)
            token_wepr = s_kj @ self.mean_weights + self.intercept

            # Sequence-level WEPR (Eq 8):
            # 1. Average of token scores S_beta
            mean_term = np.mean(token_wepr)

            # 2. Weighted sum of max contributions per rank
            # Max over tokens for each rank: (K,)
            max_contributions_per_rank = np.max(s_kj, axis=0)
            max_term = max_contributions_per_rank @ self.max_weights

            sentence_wepr = mean_term + max_term

            # Apply sigmoid to get calibrated probability of hallucination
            # P(hallucination) = sigmoid(WEPR)
            calibrated_seq_score = 1.0 / (1.0 + np.exp(-sentence_wepr))
            seq_scores.append(float(calibrated_seq_score))

            if return_per_token_scores:
                # Also apply sigmoid to token scores for consistency
                # P(token_hallucination) = sigmoid(S_beta)
                calibrated_token_scores = 1.0 / (1.0 + np.exp(-token_wepr))
                token_scores.append(calibrated_token_scores)

        return (seq_scores, token_scores) if return_per_token_scores else seq_scores
