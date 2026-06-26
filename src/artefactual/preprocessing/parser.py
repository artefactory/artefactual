"""
Module for parsing model outputs from various sources to extract log probabilities.
Each format is handled by a dedicated parser function, defined in their respective modules.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags

from artefactual.preprocessing.openai_parser import (
    is_openai_responses_api,
    process_openai_chat_completion,
    process_openai_responses_api,
    sampled_tokens_logprobs_chat_completion_api,
    sampled_tokens_logprobs_responses_api,
)
from artefactual.preprocessing.vllm_parser import (
    process_vllm_top_logprobs,
    vllm_sampled_tokens_logprobs,
)


class LogProbParser(BaseEstimator, TransformerMixin):
    """
    Wraps parse_top_logprobs as a pipeline step.

    Because this transformer accepts non-standard list[dict] inputs
    by bypassing scikit-learn validation, cross-validation
    must start from data that has already been parsed.
    """

    def __init__(self):
        pass

    def fit(self, _x, _y=None) -> "LogProbParser":
        return self

    def transform(self, x: list) -> np.ndarray:
        parsed = parse_top_logprobs(x)
        if not parsed:
            return np.empty((0, 0, 0), dtype=np.float32)

        # validation step
        for i, sample in enumerate(parsed):  # sample is the dictionary for each generation
            for token_idx, logprobs in sample.items():  # token position, ragged list of logprobs
                for rank, lp in enumerate(logprobs):  # column index or k index, logprob value
                    if lp is None or not np.isfinite(lp) or lp > 0:
                        error_msg = (
                            f"Invalid logprob at sample {i}, token {token_idx}, "
                            f"rank {rank}: {lp!r}. Expected a finite value <= 0."
                        )
                        raise ValueError(error_msg)
        max_tokens = max((max(d.keys()) + 1 for d in parsed if d), default=0)
        k = max((len(v) for d in parsed for v in d.values()), default=0)

        arr = np.full((len(parsed), max_tokens, k), np.nan, dtype=np.float32)
        for i, sample in enumerate(parsed):
            for token_idx, logprobs in sample.items():
                arr[i, token_idx, : len(logprobs)] = logprobs  # [depth, row, columns]

        return arr

    def __sklearn_tags__(self) -> Tags:
        """
        Bypasses strict scikit-learn array validation checks, allowing
        the transformer to accept a raw list of dictionaries.
        """
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.requires_fit = False  # model is stateless
        tags.input_tags.two_d_array = False
        return tags


def parse_top_logprobs(outputs: Any) -> list[dict[int, list[float]]]:
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
        return process_vllm_top_logprobs(outputs, iterations)

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


def parse_sampled_token_logprobs(outputs: Any) -> list[NDArray]:
    """
    A wrapper function to parse token probabilities from various output formats.
    First checks for vLLM format, then OpenAI ChatCompletion, and finally OpenAI Responses API.

    Args:
        outputs: Model outputs in various formats.
    Returns:
        list[NDArray]: A list of 1D numpy arrays, each containing the log probabilities
                       of the sampled tokens for one sequence.
    """
    # Check for vLLM offline inference format
    if isinstance(outputs, list) and len(outputs) > 0 and hasattr(outputs[0], "outputs"):
        if not outputs[0].outputs:
            return []
        iterations = len(outputs[0].outputs)
        return vllm_sampled_tokens_logprobs(outputs, iterations)

    # Check for OpenAI ChatCompletion format
    if hasattr(outputs, "choices") or (isinstance(outputs, dict) and "choices" in outputs):
        return sampled_tokens_logprobs_chat_completion_api(outputs)

    # Check for OpenAI Responses API format
    if is_openai_responses_api(outputs):
        return sampled_tokens_logprobs_responses_api(outputs)

    msg = (
        f"Unsupported output format: {type(outputs).__name__}. "
        "Expected vLLM RequestOutput, OpenAI ChatCompletion, or OpenAI Responses object."
    )
    raise TypeError(msg)
