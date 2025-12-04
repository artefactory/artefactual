import warnings

import numpy as np
from beartype import beartype
from numpy.typing import NDArray
from vllm import RequestOutput

from artefactual.data.data_model import Completion
from artefactual.features.data_processing import process_logprobs
from artefactual.features.entropy_contributions import compute_entropy_contributions
from artefactual.scoring.uncertainty_detector import UncertaintyDetector
from artefactual.utils.calibration import load_calibration


class EPR(UncertaintyDetector):
    """Computes Entropy Production Rate (EPR) from model completions."""

    def __init__(self, model: str | None = None, k: int = 15) -> None:
        """
        Initialize the EPR scorer.
        Args:
            model: Optional model name or path to load calibration coefficients.
                   If None, raw EPR scores are returned (uncalibrated).
            k: Number of top log probabilities to consider (default: 15).
        """
        super().__init__(k=k)
        self.intercept = 0.0
        self.coefficient = 1.0
        self.is_calibrated = False

        if model:
            try:
                calibration_data = load_calibration(model)
                self.intercept = calibration_data.get("intercept", 0.0)
                coeffs = calibration_data.get("coefficients", {})
                # EPR uses a single coefficient for the mean entropy
                self.coefficient = coeffs.get("mean_entropy", 1.0)
                self.is_calibrated = True
            except ValueError:
                warnings.warn(
                    f"Could not load calibration for model '{model}'. Proceeding with uncalibrated scores.",
                    UserWarning,
                    stacklevel=2,
                )

    @beartype
    def compute(
        self,
        outputs: list[RequestOutput],
        *,
        return_per_token_scores: bool = False,
    ) -> list[float] | tuple[list[float], list[NDArray[np.floating]]]:
        """
        Compute EPR-based uncertainty scores from a sequence of completions.
        Args:
            completions: A list of completions, where each completion is a list of tokens,
                         and each token is a list of its top-K log probabilities.
                         Shape: (num_completions, num_tokens, num_logprobs)
        Returns:
            - List of sequence-level EPR scores.
            - Optionally (if return_per_token_scores=True),
              a tuple of (seq_scores, per_token_scores).
        """
        if not outputs:
            return []

        seq_scores: list[float] = []
        token_scores: list[NDArray[np.floating]] = []

        iterations = len(outputs[0].outputs)
        completions_data = process_logprobs(outputs, iterations)
        completions = [Completion(token_logprobs=data) for data in completions_data]

        for completion in completions:
            token_logprobs_dict = completion.token_logprobs
            if not token_logprobs_dict:
                # If no tokens, apply calibration logic as in main block
                if self.is_calibrated:
                    linear_score = self.coefficient * 0.0 + self.intercept
                    score = 1.0 / (1.0 + np.exp(-linear_score))
                else:
                    score = 0.0
                seq_scores.append(score)

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

            # Token-level EPR: sum across K
            token_epr = s_kj.sum(axis=1)  # shape (num_tokens,)

            # Sequence-level EPR: mean of tokens
            seq_epr = float(token_epr.mean()) if token_epr.size > 0 else 0.0

            # Apply calibration if available
            if self.is_calibrated:
                # Linear transformation: alpha * EPR + beta
                linear_score = self.coefficient * seq_epr + self.intercept
                # Sigmoid calibration: P(hallucination) = sigmoid(linear_score)
                calibrated_score = 1.0 / (1.0 + np.exp(-linear_score))
                seq_scores.append(float(calibrated_score))
            else:
                seq_scores.append(seq_epr)

            if return_per_token_scores:
                token_scores.append(token_epr)

        if return_per_token_scores:
            return seq_scores, token_scores
        return seq_scores
