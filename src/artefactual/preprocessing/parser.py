"""Extraction of top-k logprobs from completion responses into a dense array.

Responses are validated against the models in `response_models` and dispatched by type,
so a new provider format is added by registering an extractor rather than by editing the
entry points here.
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
    return process_openai_chat_completion(response)


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
    """First pipeline step: completion responses in, a dense logprob array out.

    Accepts raw responses rather than arrays, so it opts out of scikit-learn's input
    validation. Cross-validation over a `BaseDetector` therefore has to start from data
    this step has already parsed, since sklearn cannot index a response object by row.

    This step owns the rank axis. It sizes the output to `k` and refuses responses that
    carry fewer ranks, which is possible only here: downstream, every token has been padded
    to a common width and a token's original rank count is no longer recoverable.

    Args:
        k: Rank count to emit. `None` uses the widest rank count present in the batch.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k

    def fit(self, X, y=None) -> "LogProbParser":  # noqa: ARG002, N803 — X / y unused but required by the sklearn fit signature
        """No-op, present so the step composes in a `Pipeline`. Returns self."""
        return self

    def transform(self, X: list) -> np.ndarray:  # noqa: N803
        """Parse responses into a NaN-padded logprob array.

        Args:
            X: One completion response, or a sequence of them. Each yields one row per
                sampled sequence.

        Returns:
            `(n_sequences, max_tokens, k)`, NaN where a sequence is shorter than the
            longest in the batch or a token position carried no ranks. Empty input gives
            an empty `(0, 0, 0)` array.

        Raises:
            TypeError: If a response is not a recognised completion format.
            ValueError: If a logprob is missing, non-finite or positive, if no response
                carries any log-probabilities, or if a response carries fewer than `k`
                ranks per token.
        """
        parsed = parse_top_logprobs(X)
        if not parsed:
            return np.empty((0, 0, 0), dtype=np.float64)

        # Rejected before the array is built: NaN is the padding marker downstream, so an
        # invalid logprob cast into the array would be indistinguishable from padding.
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
        if max_tokens == 0 and self.k is not None:
            # The shape was recognised but nothing carried ranks, which is what a provider
            # returns when logprobs were not requested or are not supported. Caught here
            # because a (n, 0, k) array reaches the entropy step as a zero-size reduction,
            # whose numpy error names nothing the caller can act on. Gated on `k` for the
            # same reason `_reject_narrow` is: without a declared rank count the parser has
            # no expectation to contradict, and an empty parse is just an empty parse.
            error_msg = (
                f"None of the {len(parsed)} response(s) carry any log-probabilities. The "
                f"payload is a recognised completion format but its top_logprobs are "
                f"absent or empty, which is what a provider returns when logprobs were "
                f"not requested or are unsupported. Regenerate with logprobs=True and "
                f"top_logprobs={self.k if self.k is not None else 'k'}."
            )
            raise ValueError(error_msg)
        self._reject_narrow(parsed)

        # float64: `LogisticRegression` takes the dtype of the features it is fitted on, and
        # the pretrained coefficients are read from JSON at full decimal precision. Matching
        # them keeps a fitted detector and a pretrained one on one dtype, and keeps the dot
        # product from upcasting on every call.
        #
        # Surplus ranks are dropped rather than kept: a calibration fit at k has no
        # coefficient for rank k+1, and EPR's mean is defined over exactly k ranks.
        width = self.k if self.k is not None else max((len(v) for d in parsed for v in d.values()), default=0)

        arr = np.full((len(parsed), max_tokens, width), np.nan, dtype=np.float64)
        for i, sample in enumerate(parsed):
            for token_idx, logprobs in sample.items():
                ranks = logprobs[:width]
                arr[i, token_idx, : len(ranks)] = ranks  # [depth, row, columns]

        return arr

    def _reject_narrow(self, parsed: list[dict[int, list[float]]]) -> None:
        """Raise if any response carries fewer than `k` ranks per token.

        Compared per response, not across the batch, because padding is applied per batch:
        a wide response would otherwise raise the measured width above a narrow sibling's
        own rank count.

        Token positions holding no ranks are skipped. Those are the empty-`top_logprobs`
        case and remain padding, which the entropy step reports as an empty sequence.
        """
        if self.k is None:
            return

        for i, sample in enumerate(parsed):
            widths = [len(logprobs) for logprobs in sample.values() if logprobs]
            if widths and min(widths) < self.k:
                error_msg = (
                    f"Response {i} carries {min(widths)} rank(s) per token but k={self.k} "
                    f"was requested. The missing ranks are not absent from the "
                    f"distribution, only unfetched, so zero-filling them would drop their "
                    f"entropy contributions and score the response as more confident than "
                    f"it was. Regenerate with top_logprobs={self.k}, or score at "
                    f"k={min(widths)} with a detector trained at that rank count."
                )
                raise ValueError(error_msg)

    def __sklearn_tags__(self) -> Tags:
        """Declares the step stateless and exempt from array validation."""
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.requires_fit = False  # model is stateless
        tags.input_tags.two_d_array = False
        return tags


def parse_top_logprobs(outputs: Any) -> list[dict[int, list[float]]]:
    """Extract each token's top-k logprobs, one entry per sampled sequence.

    Args:
        outputs: A completion response, as a mapping or an object, or a sequence of them
            parsed and concatenated in order.

    Returns:
        One dict per sequence, mapping token position to that token's ranks. Ranks are
        ragged: a position whose `top_logprobs` was empty maps to an empty list, and the
        position is still counted so token indices track the generated text.

    Raises:
        TypeError: If the output is not a recognised completion format.
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
    """Extract the logprob of each *sampled* token, ignoring the alternatives.

    The counterpart to `parse_top_logprobs`, which reads the top-k distribution. Neither
    EPR nor WEPR uses this; it is exposed for callers measuring sequence likelihood.

    Args:
        outputs: A single completion response, as a mapping or an object.

    Returns:
        One 1-D array per sequence, holding the sampled token logprobs in order.

    Raises:
        TypeError: If the output is not a recognised completion format.
    """
    try:
        response = _RESPONSE_ADAPTER.validate_python(outputs, from_attributes=True)
    except ValidationError as error:
        msg = f"Unsupported output format: {type(outputs).__name__}. Expected a completion response carrying logprobs."
        raise TypeError(msg) from error

    return _sampled_logprobs(response)
