from typing import Any

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from artefactual.data.data_model import Completion
from artefactual.scoring.entropy_methods.entropy_contributions import compute_entropy_contributions
from artefactual.scoring.entropy_methods.uncertainty_detector import UncertaintyDetector
from artefactual.utils.io import load_weights


class WEPR(UncertaintyDetector):
    def __init__(self, model: str) -> None:
        """
        Initialize the WEPR scorer with weights loaded from the specified source.
        Args:
            model: Either a built-in model name or a local file path to load weights from.
        """
        weights_data = load_weights(model)
        self.intercept = float(weights_data.get("intercept", 0.0))
        coeffs_raw = weights_data.get("coefficients", {})
        coeffs: dict[str, float] = coeffs_raw if isinstance(coeffs_raw, dict) else {}

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

    def _compute_impl(
        self,
        outputs: Any,
    ) -> tuple[list[float], list[NDArray[np.floating]]]:
        """
        Internal implementation to compute WEPR scores.
        """
        if not outputs:
            return [], []

        completions_data = self._parse_outputs(outputs)
        completions = [Completion(token_logprobs=data) for data in completions_data]

        seq_scores: list[float] = []
        token_scores: list[NDArray[np.floating]] = []

        for completion in completions:
            token_logprobs_dict = completion.token_logprobs
            if not token_logprobs_dict:
                # If no tokens, return the calibrated baseline probability
                baseline_prob = 1.0 / (1.0 + np.exp(-self.intercept))  # Sigmoid of intercept
                seq_scores.append(baseline_prob)
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

            # Also apply sigmoid to token scores for consistency
            # P(token_hallucination) = sigmoid(S_beta)
            calibrated_token_scores = 1.0 / (1.0 + np.exp(-token_wepr))
            token_scores.append(calibrated_token_scores)

        return seq_scores, token_scores

    @beartype
    def compute(self, outputs: Any) -> list[float]:
        """
        Compute WEPR-based uncertainty scores from a sequence of completions.
        Args:
            outputs: Model outputs. Can be:
                     - List of vLLM RequestOutput objects.
                     - OpenAI ChatCompletion object (or dict).
                     - OpenAI Responses object (or dict).
        Returns:
            List of sequence-level WEPR scores.
        """
        return self._compute_impl(outputs)[0]

    @beartype
    def compute_token_scores(self, outputs: Any) -> list[NDArray[np.floating]]:
        """
        Compute token-level WEPR scores from a sequence of completions.
        Args:
            outputs: Model outputs. Can be:
                     - List of vLLM RequestOutput objects.
                     - OpenAI ChatCompletion object (or dict).
                     - OpenAI Responses object (or dict).
        Returns:
            List of token-level WEPR scores (numpy arrays).
        """
        return self._compute_impl(outputs)[1]
