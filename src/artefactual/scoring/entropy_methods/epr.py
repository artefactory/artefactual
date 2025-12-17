from typing import Any, cast

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

    def __init__(self, pretrained_model_name_or_path: str | None = None, k: int = 15) -> None:
        """
        Initialize the EPR scorer.

        Args:
            pretrained_model_name_or_path: Model name or path to load calibration coefficients.
                   This argument is required — an error is raised if it is missing or invalid.
            k: Number of top log probabilities to consider (default: 15).
        Raises:
            ValueError: If `pretrained_model_name_or_path` is not provided or calibration cannot be loaded.
        """
        super().__init__(k=k)
        self.intercept = 0.0
        self.coefficient = 1.0
        self.is_calibrated = False

        if pretrained_model_name_or_path is None:
            msg = "pretrained_model_name_or_path is required"
            raise ValueError(msg)

        try:
            calibration_data: dict[str, Any] = load_calibration(pretrained_model_name_or_path)
        except Exception as exc:
            msg = f"Failed to load calibration for '{pretrained_model_name_or_path}': {exc}"
            raise ValueError(msg) from exc

        self.intercept = calibration_data.get("intercept", 0.0)
        coeffs: dict[str, float] = cast(dict[str, float], calibration_data.get("coefficients", {}))
        self.coefficient = coeffs.get("mean_entropy", 1.0)
        self.is_calibrated = True

    def _compute_impl(
        self,
        outputs: Any,
    ) -> tuple[list[float], list[NDArray[np.floating]]]:
        """
        Internal implementation to compute EPR scores.

        Args:
            outputs: Model outputs. Can be:
                     - List of vLLM RequestOutput objects.
                     - OpenAI ChatCompletion object (or dict).
                     - OpenAI Responses object (or dict).

        Returns:
            A tuple containing:
            - List of sequence-level EPR scores.
            - List of token-level EPR scores (numpy arrays).
        """
        if not outputs:
            return [], []

        completions_data = self._parse_outputs(outputs)

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

            # sum over rank K (Token EPR)
            token_epr = np.sum(s_kj, axis=1)  # shape = (num_tokens_in_sequence)

            # Mean over sequence (Sequence EPR)
            # Make sure to cast to float !
            seq_epr = float(np.mean(token_epr)) if token_epr.size > 0 else 0.0

            seq_scores.append(self._apply_calibration(seq_epr))

            token_scores.append(self._apply_calibration(token_epr))

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
        """
        Returns the default score (calibrated or not).

        Returns:
            The default score.
        """
        if self.is_calibrated:
            # sigmoid(intercept)
            return 1.0 / (1.0 + np.exp(-self.intercept))
        return 0.0

    def _apply_calibration(self, raw_epr: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
        """
        Apply logistic calibration to a raw EPR score or array of scores.
        The calibration uses a linear transformation followed by a sigmoid:
            calibrated = sigmoid(coefficient * raw_epr + intercept)
        If ``self.is_calibrated`` is ``False``, the input is returned unchanged.
        Args:
            raw_epr: Raw EPR value(s). May be a single float (sequence-level score)
                or a NumPy array of floats (token-level scores). NumPy broadcasting
                is used so the calibration is applied element-wise to arrays.
        Returns:
            A calibrated score with the same shape as ``raw_epr``:
            - If ``raw_epr`` is a float, returns a single calibrated float.
            - If ``raw_epr`` is an NDArray, returns an NDArray of calibrated scores.
        """
        if self.is_calibrated:
            linear_score = self.coefficient * raw_epr + self.intercept
            return 1.0 / (1.0 + np.exp(-linear_score))
        return raw_epr
