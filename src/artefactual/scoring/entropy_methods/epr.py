import warnings
from typing import Any

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from artefactual.data.data_model import Completion
from artefactual.scoring.entropy_methods.entropy_contributions import compute_entropy_contributions
from artefactual.scoring.entropy_methods.uncertainty_detector import UncertaintyDetector
from artefactual.utils.io import load_calibration


class EPR(UncertaintyDetector):
    """
    Computes Entropy Production Rate (EPR) from model completions.

    Supports:
    - vLLM (RequestOutput)
    - OpenAI Chat Completions (classic 'choices' format)
    - OpenAI Responses API (new 'output' format)
    """

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
                self.intercept = float(calibration_data.get("intercept", 0.0))
                coeffs_raw = calibration_data.get("coefficients", {})
                coeffs: dict[str, float] = coeffs_raw if isinstance(coeffs_raw, dict) else {}
                self.coefficient = float(coeffs.get("mean_entropy", 1.0))
                self.is_calibrated = True
            except ValueError:
                warnings.warn(
                    f"Could not load calibration for model '{model}'. Proceeding with uncalibrated scores.",
                    UserWarning,
                    stacklevel=2,
                )

    def _compute_impl(
        self,
        outputs: Any,
    ) -> tuple[list[float], list[NDArray[np.floating]]]:
        """
        Internal implementation to compute EPR scores.
        """
        if not outputs:
            return [], []

        completions_data = self._parse_outputs(outputs)

        # --- 2. COMPUTATION (Logic Core) ---

        completions = [Completion(token_logprobs=data) for data in completions_data]
        seq_scores: list[float] = []
        token_scores: list[NDArray[np.floating]] = []

        for completion in completions:
            token_logprobs_dict = completion.token_logprobs

            # Handle empty case
            if not token_logprobs_dict:
                seq_scores.append(self._get_default_score())
                token_scores.append(np.array([], dtype=np.float32))
                continue

            # Prepare vectorized data
            sorted_indices = sorted(token_logprobs_dict.keys())
            logprobs_list = [token_logprobs_dict[i] for i in sorted_indices]

            # Vectorized Entropy Calculation
            s_kj = compute_entropy_contributions(logprobs_list, self.k)

            # Sum over K (Token EPR)
            token_epr = s_kj.sum(axis=1)

            # Mean over sequence (Sequence EPR)
            # Make sure to cast to float !
            seq_epr = float(token_epr.mean()) if token_epr.size > 0 else 0.0

            seq_scores.append(self._apply_calibration(seq_epr))

            token_scores.append(token_epr)

        return seq_scores, token_scores

    @beartype
    def compute(self, outputs: Any) -> list[float]:
        """
        Compute EPR-based uncertainty scores from a sequence of completions.
        Args:
            outputs: Model outputs. Can be:
                     - List of vLLM RequestOutput objects.
                     - OpenAI ChatCompletion object (or dict).
                     - OpenAI Responses object (or dict).
        Returns:
            List of sequence-level EPR scores.
        """
        return self._compute_impl(outputs)[0]

    @beartype
    def compute_token_scores(self, outputs: Any) -> list[NDArray[np.floating]]:
        """
        Compute token-level EPR scores from a sequence of completions.
        Args:
            outputs: Model outputs. Can be:
                     - List of vLLM RequestOutput objects.
                     - OpenAI ChatCompletion object (or dict).
                     - OpenAI Responses object (or dict).
        Returns:
            List of token-level EPR scores (numpy arrays).
        """
        return self._compute_impl(outputs)[1]

    def _get_default_score(self) -> float:
        """Returns the default score (calibrated or not)."""
        if self.is_calibrated:
            # sigmoid(intercept)
            return 1.0 / (1.0 + np.exp(-self.intercept))
        return 0.0

    def _apply_calibration(self, raw_epr: float) -> float:
        """Applies logistic calibration."""
        if self.is_calibrated:
            linear_score = self.coefficient * raw_epr + self.intercept
            return 1.0 / (1.0 + np.exp(-linear_score))
        return raw_epr
