from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from artefactual.preprocessing.openai_parser import (
    is_openai_responses_api,
    process_openai_chat_completion,
    process_openai_responses_api,
)
from artefactual.preprocessing.vllm_parser import process_vllm_logprobs


class UncertaintyDetector(ABC):
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

    @abstractmethod
    def compute(self, inputs: Any) -> list[float]:
        """
        Compute sequence-level uncertainty scores from inputs.

        Args:
            inputs: The inputs to process (e.g. completions or model outputs).

        Returns:
            The computed sequence-level scores.
        """
        pass

    @abstractmethod
    def compute_token_scores(self, inputs: Any) -> list[NDArray[np.floating]]:
        """
        Compute token-level uncertainty scores from inputs.

        Args:
            inputs: The inputs to process (e.g. completions or model outputs).

        Returns:
            The computed token-level scores.
        """
        pass

    @staticmethod
    def _parse_outputs(outputs: Any) -> list[dict[int, list[float]]]:
        """Parse different output formats to extract logprobs."""
        # vLLM parser
        if isinstance(outputs, list) and len(outputs) > 0 and hasattr(outputs[0], "outputs"):
            return process_vllm_logprobs(outputs[0].outputs)

        # B. OpenAI parser for classic ChatCompletion
        if hasattr(outputs, "choices") or (isinstance(outputs, dict) and "choices" in outputs):
            choices = outputs.choices if hasattr(outputs, "choices") else outputs["choices"]
            return process_openai_chat_completion(outputs, iterations=len(choices))

        # C. OpenAI parser for Responses API
        if is_openai_responses_api(outputs):
            return process_openai_responses_api(outputs)

        msg = (
            f"Unsupported output format: {type(outputs).__name__}. "
            "Expected vLLM RequestOutput, OpenAI ChatCompletion, or OpenAI Responses object."
        )
        raise TypeError(msg)
