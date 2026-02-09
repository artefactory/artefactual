from typing import cast

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from artefactual.data.data_model import Completion
from artefactual.scoring.entropy_methods.entropy_contributions import compute_entropy_contributions
from artefactual.scoring.uncertainty_detector import LogProbUncertaintyDetector
from artefactual.utils.io import load_weights


class WEPR(LogProbUncertaintyDetector):
    """
    Computes Weighted Entropy Production Rate (WEPR) from model log probabilities.
    WEPR extends EPR by applying learned weights to the entropy contributions based on their ranks.
    It computes both mean-weighted and max-weighted contributions to produce a sequence-level uncertainty score.
    Token-level WEPR scores are also provided.
    You can parse raw model outputs using the `parse_top_logprobs` method from `artefactual.preprocessing`.
    """

    def __init__(self, pretrained_model_name_or_path: str) -> None:
        """
        Initialize the WEPR scorer with weights loaded from the specified source.

        Args:
            pretrained_model_name_or_path: Either a built-in model name or a local file path to load weights from.
        """
        weights_data = load_weights(pretrained_model_name_or_path)
        self.intercept = weights_data.get("intercept", 0.0)
        coeffs = cast(dict[str, float], weights_data.get("coefficients", {}))

        # Determine k from the coefficients (assuming keys like "mean_rank_15")
        # We look for the maximum rank index present in the coefficients
        ranks = [
            int(key.split("_")[-1]) for key in coeffs if key.startswith("mean_rank_") and key.split("_")[-1].isdigit()
        ]
        k = max(ranks) if ranks else 15

        super().__init__(k=k)

        # Parse weights into numpy arrays for vectorized computation
        self.mean_weights = np.zeros(k, dtype=np.float32)
        self.max_weights = np.zeros(k, dtype=np.float32)

        for i in range(1, k + 1):
            self.mean_weights[i - 1] = coeffs.get(f"mean_rank_{i}", 0.0)
            self.max_weights[i - 1] = coeffs.get(f"max_rank_{i}", 0.0)

    def _compute_wepr_stats(
        self,
        parsed_logprobs: list[dict[int, list[float]]],
    ) -> list[tuple[NDArray[np.floating], NDArray[np.floating] | None]]:
        """
        Internal implementation to compute WEPR intermediate statistics.

        Args:
            parsed_logprobs: Parsed log probabilities.

        Returns:
            A list of tuples containing:
            - Token-level WEPR scores (S_beta) (numpy array).
            - Max contributions per rank (numpy array of shape (K,)), or None if empty.
        """
        if not parsed_logprobs:
            return []

        completions = [Completion(token_logprobs=data) for data in parsed_logprobs]
        stats: list[tuple[NDArray[np.floating], NDArray[np.floating] | None]] = []

        for completion in completions:
            token_logprobs_dict = completion.token_logprobs
            if not token_logprobs_dict:
                # Empty completion
                stats.append((np.array([], dtype=np.float32), None))
                continue

            # Convert to a 2D numpy array for vectorized processing
            # Sort by token position to ensure correct order
            sorted_indices = sorted(token_logprobs_dict.keys())
            logprobs_list = [token_logprobs_dict[i] for i in sorted_indices]

            # Compute entropy contributions in a vectorized manner
            # Input shape: (num_tokens_in_sequence, K)
            s_kj = compute_entropy_contributions(logprobs_list, self.k)

            # Token-level WEPR (S_beta): weighted sum across K using mean_weights
            # S_beta = sum(beta_k * s_kj) + beta_0
            token_wepr = s_kj @ self.mean_weights + self.intercept

            # Max over tokens for each rank: (K,)
            max_contributions_per_rank = np.max(s_kj, axis=0)

            stats.append((token_wepr, max_contributions_per_rank))

        return stats

    @beartype
    def compute(self, parsed_logprobs: list[dict[int, list[float]]]) -> list[float]:
        """
        Compute WEPR-based uncertainty scores from parsed log probabilities.
        You can parse raw model outputs using the `parse_top_logprobs` method from `artefactual.preprocessing`.

        Args:
            parsed_logprobs: Parsed log probabilities.

        Returns:
            List of sequence-level WEPR scores.
        """
        stats = self._compute_wepr_stats(parsed_logprobs)
        seq_scores: list[float] = []

        for token_wepr, max_contributions in stats:
            if max_contributions is None:
                # If no tokens, return the calibrated baseline probability
                baseline_prob = 1.0 / (1.0 + np.exp(-self.intercept))
                seq_scores.append(baseline_prob)
                continue

            # Sequence-level WEPR (Eq 8):
            # 1. Average of token scores S_beta
            mean_term = np.mean(token_wepr)

            # 2. Weighted sum of max contributions per rank
            max_term = max_contributions @ self.max_weights

            sentence_wepr = mean_term + max_term

            # Apply sigmoid to get calibrated probability of hallucination
            # P(hallucination) = sigmoid(WEPR)
            calibrated_seq_score = 1.0 / (1.0 + np.exp(-sentence_wepr))
            seq_scores.append(float(calibrated_seq_score))

        return seq_scores

    @beartype
    def compute_token_scores(self, parsed_logprobs: list[dict[int, list[float]]]) -> list[NDArray[np.floating]]:
        """
        Compute token-level WEPR scores from parsed logprobs.
        You can parse raw model outputs using the `parse_top_logprobs` method from `artefactual.preprocessing`.

        Args:
            parsed_logprobs: Parsed log probabilities.

        Returns:
            List of token-level WEPR scores (numpy arrays).
        """
        stats = self._compute_wepr_stats(parsed_logprobs)
        token_scores: list[NDArray[np.floating]] = []

        for token_wepr, _ in stats:
            # Apply sigmoid to token scores for consistency
            # P(token_hallucination) = sigmoid(S_beta)
            if token_wepr.size > 0:
                calibrated_token_scores = 1.0 / (1.0 + np.exp(-token_wepr))
                token_scores.append(calibrated_token_scores)
            else:
                token_scores.append(token_wepr)

        return token_scores
