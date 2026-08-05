"""
Module for parsing model outputs from various sources to extract log probabilities.
Each format is handled by a dedicated parser function, defined in their respective modules.
"""

from functools import singledispatch
from typing import Any

import numpy as np
from beartype.door import is_bearable
from pydantic import TypeAdapter, ValidationError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags

from artefactual.preprocessing.openai_parser import (
    process_openai_chat_completion,
    process_openai_responses_api,
    sampled_tokens_logprobs_chat_completion_api,
    sampled_tokens_logprobs_responses_api,
)
from artefactual.preprocessing.response_models import ChatCompletion, ResponsesPayload

_RESPONSE_ADAPTER = TypeAdapter(ChatCompletion | ResponsesPayload)


@singledispatch
def _top_logprobs(response: Any) -> list[dict[int, list[float]]]:
    """Extract per-token top logprobs from a validated response. Register one per format."""
    msg = f"No top-logprob extractor registered for {type(response).__name__}."
    raise TypeError(msg)


@_top_logprobs.register
def _(response: ChatCompletion) -> list[dict[int, list[float]]]:
    return process_openai_chat_completion(response, iterations=len(response.choices))


@_top_logprobs.register
def _(response: ResponsesPayload) -> list[dict[int, list[float]]]:
    return process_openai_responses_api(response)


@singledispatch
def _sampled_logprobs(response: Any) -> list[np.ndarray]:
    """Extract sampled-token logprobs from a validated response. Register one per format."""
    msg = f"No sampled-logprob extractor registered for {type(response).__name__}."
    raise TypeError(msg)


@_sampled_logprobs.register
def _(response: ChatCompletion) -> list[np.ndarray]:
    return sampled_tokens_logprobs_chat_completion_api(response)


@_sampled_logprobs.register
def _(response: ResponsesPayload) -> list[np.ndarray]:
    return sampled_tokens_logprobs_responses_api(response)


class LogProbParser(BaseEstimator, TransformerMixin):
    """
    Wraps parse_top_logprobs as a pipeline step.

    Because this transformer accepts non-standard list[dict] inputs
    by bypassing scikit-learn validation, cross-validation
    must start from data that has already been parsed.
    """

    def fit(self, X, y=None) -> "LogProbParser":  # noqa: ARG002, N803 — X / y unused but required by the sklearn fit signature
        return self

    def transform(self, X: list) -> np.ndarray:  # noqa: N803
        parsed = parse_top_logprobs(X)
        if not parsed:
            return np.empty((0, 0, 0), dtype=np.float32)

        # validation step
        # TODO(perf): O(n·T·k) Python loop — vectorize over the padded array once batches get large.
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
                    - A completion response, as a mapping or an object.
                    - A sequence of responses, parsed and concatenated in order.

    Returns:
        List of dictionaries mapping token indices to lists of log probs,
        one per generated sequence.

    Raises:
        TypeError: If the output format is not supported.
    """
    if is_bearable(outputs, list | tuple):
        return [sequence for response in outputs for sequence in parse_top_logprobs(response)]

    try:
        response = _RESPONSE_ADAPTER.validate_python(outputs, from_attributes=True)
    except ValidationError as error:
        msg = f"Unsupported output format: {type(outputs).__name__}. Expected a completion response carrying logprobs."
        raise TypeError(msg) from error

    return _top_logprobs(response)


def parse_sampled_token_logprobs(outputs: Any) -> list[np.ndarray]:
    """
    A wrapper function to parse token probabilities from various output formats.
    Handles the OpenAI ChatCompletion and OpenAI Responses API shapes.

    Args:
        outputs: Model outputs in various formats.
    Returns:
        list[np.ndarray]: A list of 1D numpy arrays, each containing the log probabilities
                       of the sampled tokens for one sequence.
    """
    try:
        response = _RESPONSE_ADAPTER.validate_python(outputs, from_attributes=True)
    except ValidationError as error:
        msg = f"Unsupported output format: {type(outputs).__name__}. Expected a completion response carrying logprobs."
        raise TypeError(msg) from error

    return _sampled_logprobs(response)
