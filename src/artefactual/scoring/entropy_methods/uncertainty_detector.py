from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from artefactual.preprocessing import (
    is_openai_responses_api,
    process_openai_chat_completion,
    process_openai_responses_api,
    process_vllm_logprobs,
)


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

    @staticmethod
    def _parse_outputs(outputs: Any) -> list[dict[int, list[float]]]:
        """
        Parse different output formats to extract logprobs.

        Args:
            outputs: Model outputs. Can be:
                     - List of vLLM RequestOutput objects.
                     - OpenAI ChatCompletion object (or dict).
                     - OpenAI Responses object (or dict).

        Returns:
            List of dictionaries mapping token indices to lists of log probs.

        Raises:
            TypeError: If the output format is not supported.
        """
        # vLLM parser
        if isinstance(outputs, list) and len(outputs) > 0 and hasattr(outputs[0], "outputs"):
            if not outputs[0].outputs:
                return []
            iterations = len(outputs[0].outputs)
            return process_vllm_logprobs(outputs, iterations)

        # OpenAI parser for classic ChatCompletion
        if hasattr(outputs, "choices") or (isinstance(outputs, dict) and "choices" in outputs):
            choices = outputs.choices if hasattr(outputs, "choices") else outputs["choices"]
            return process_openai_chat_completion(outputs, iterations=len(choices))

        # OpenAI parser for Responses API
        if is_openai_responses_api(outputs):
            return process_openai_responses_api(outputs)

        msg = (
            f"Unsupported output format: {type(outputs).__name__}. "
            "Expected vLLM RequestOutput, OpenAI ChatCompletion, or OpenAI Responses object."
        )
        raise TypeError(msg)
