"""Extraction of top-k logprobs from completion responses into a dense array.

Responses are validated against the models in `response_models`, which carry the
extraction themselves, so a new provider format is a new model there rather than an edit
to the entry points here.
"""

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
from beartype.door import is_bearable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags

from artefactual.preprocessing.response_models import LogprobsMixin


# Base order is significant. `__sklearn_tags__` resolves through `super()`, so
# `TransformerMixin` must precede `BaseEstimator` for the mixin's override to run. In the
# reverse order `transformer_tags` stays `None` and scikit-learn does not treat the class
# as a transformer.
class LogProbParser(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):
    """First pipeline step: completion responses in, a dense logprob array out.

    Accepts raw responses rather than arrays, so it opts out of scikit-learn's input
    validation. Cross-validation over a `BaseDetector` therefore has to start from data
    this step has already parsed, since sklearn cannot index a response object by row.
    A numeric array is refused with a `TypeError`.

    This step owns the rank axis -- a token's candidates ordered by likelihood, one column
    per rank. It sizes the output to `k` and refuses responses carrying fewer candidates,
    which is possible only here: downstream, every token has been padded to a common width
    and a token's original candidate count is no longer recoverable.

    Parameters
    ----------
    k : int, default=None
        Number of candidates to emit per token, and so the width of the rank axis. `None`
        uses the widest present in the batch.

    Examples
    --------
    >>> cube = LogProbParser(k=15).transform(responses)  # doctest: +SKIP
    >>> cube.shape  # (n_sequences, max_tokens, 15)  # doctest: +SKIP
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k

    def set_output(self, *, transform: str | None = None) -> "LogProbParser":  # noqa: ARG002 — accepted so `Pipeline.set_output` can reach the steps after it
        """Accepted and ignored: the `(n, tokens, k)` cube has no table form to convert to.

        Output wrapping is off for this step. `Pipeline.set_output` calls every step, and the
        entropy step after this one is the first that yields a table.

        Parameters
        ----------
        transform : str, default=None
            Ignored.

        Returns
        -------
        self : LogProbParser
        """
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the rank columns, which the entropy step maps on.

        Parameters
        ----------
        input_features : array-like of str, default=None
            If given, its first `k` entries name the columns instead of `rank0`, ...

        Returns
        -------
        ndarray of str
            One name per rank column.

        Raises
        ------
        ValueError
            If `k` is `None` and no `input_features` are given, since the width is then
            only known from data.
        """
        if input_features is not None:
            return np.asarray(input_features, dtype=object)[: self.k]
        if self.k is None:
            msg = "The width is only known from data when k is None; set k to name the rank columns."
            raise ValueError(msg)
        return np.array([f"rank{i}" for i in range(self.k)], dtype=object)

    def fit(self, X, y=None) -> "LogProbParser":  # noqa: ARG002, N803 — X / y unused but required by the sklearn fit signature
        """No-op, present so the step composes in a `Pipeline`.

        Parameters
        ----------
        X : list
            Ignored.
        y : array-like, default=None
            Ignored.

        Returns
        -------
        self : LogProbParser
        """
        return self

    def transform(self, X: list) -> np.ndarray:  # noqa: N803
        """Parse responses into a NaN-padded logprob array.

        Parameters
        ----------
        X : list
            One completion response, or a sequence of them. Each yields one row per
            sampled sequence.

        Returns
        -------
        ndarray of shape (n_sequences, max_tokens, k)
            NaN where a sequence is shorter than the longest in the batch or a token
            position carried no candidates. Empty input gives an empty `(0, 0, 0)` array.

        Raises
        ------
        TypeError
            If a response is not a recognised completion format, including a batch line
            whose body is not one.
        ValueError
            If a logprob is missing, non-finite or positive, if no response carries any
            log-probabilities, if a response carries fewer than `k` candidates per token, or
            if a batch line carries no completion at all.
        """
        parsed = parse_top_logprobs(X)
        if not parsed:
            return np.empty((0, 0, 0), dtype=np.float64)

        # Rejected before the array is built: NaN is the padding marker downstream, so an
        # invalid logprob cast into the array would be indistinguishable from padding.
        # TODO(perf): O(n·T·k) Python loop — vectorize over the padded array once batches get large.
        for i, sample in enumerate(parsed):  # sample is the dictionary for each generation
            for token_idx, logprobs in sample.items():  # token position, ragged list of logprobs
                for rank, lp in enumerate(logprobs):  # rank (column index), that candidate's logprob
                    if lp is None or not np.isfinite(lp) or lp > 0:
                        error_msg = (
                            f"Invalid logprob at sample {i}, token {token_idx}, "
                            f"rank {rank}: {lp!r}. Expected a finite value <= 0."
                        )
                        raise ValueError(error_msg)
        max_tokens = max((max(d, default=-1) + 1 for d in parsed), default=0)
        if max_tokens == 0 and self.k is not None:
            # The shape was recognised but nothing carried candidates, which is what a provider
            # returns when logprobs were not requested or are not supported. Caught here
            # because a (n, 0, k) array reaches the entropy step as a zero-size reduction,
            # whose numpy error names nothing the caller can act on. Gated on `k` for the
            # same reason the width check below is: without a declared candidate count the parser has
            # no expectation to contradict, and an empty parse is just an empty parse.
            error_msg = (
                f"None of the {len(parsed)} response(s) carry any log-probabilities. The "
                f"payload is a recognised completion format but its top_logprobs are "
                f"absent or empty, which is what a provider returns when logprobs were "
                f"not requested or are unsupported. Regenerate with logprobs=True and "
                f"top_logprobs={self.k}."
            )
            raise ValueError(error_msg)
        # Narrow responses are refused per response, not across the batch, because padding is
        # applied per batch: a wide response would otherwise raise the measured width above a
        # narrow sibling's own candidate count. Token positions holding no candidates are
        # skipped: they are the empty-`top_logprobs` case and remain padding, which the
        # entropy step reports as an empty sequence.
        k = self.k
        if k is not None:
            for i, sample in enumerate(parsed):
                narrowest = min(map(len, filter(None, sample.values())), default=k)
                if narrowest < k:
                    error_msg = (
                        f"Response {i} carries {narrowest} candidate(s) per token but k={k} "
                        f"was requested. The missing candidates are not absent from the "
                        f"distribution, only unfetched, so zero-filling them would drop their "
                        f"entropy contributions and score the response as more confident than "
                        f"it was. Regenerate with top_logprobs={k}, or score at "
                        f"k={narrowest} with a detector trained at that width."
                    )
                    raise ValueError(error_msg)

        # float64: `LogisticRegression` takes the dtype of the features it is fitted on, and
        # the pretrained coefficients are stored at full float64 precision. Matching
        # them keeps a fitted detector and a pretrained one on one dtype, and keeps the dot
        # product from upcasting on every call.
        #
        # Surplus candidates are dropped rather than kept: a calibration fit at k has no
        # coefficient for rank k (0-based), and EPR's mean is defined over exactly k ranks.
        width = self.k if self.k is not None else max((len(v) for d in parsed for v in d.values()), default=0)

        arr = np.full((len(parsed), max_tokens, width), np.nan, dtype=np.float64)
        for i, sample in enumerate(parsed):
            for token_idx, logprobs in sample.items():
                candidates = logprobs[:width]
                arr[i, token_idx, : len(candidates)] = candidates  # [depth, row, columns]

        return arr

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
        outputs: A completion response, as a mapping or an object, or a collection of them
            (list, tuple, 1-D object array, Series) parsed and concatenated in order.

    Returns:
        One dict per sequence, mapping token position to that token's candidate
        log-probabilities, most likely first. Ragged: a position whose `top_logprobs` was
        empty maps to an empty list, and the position is still counted so token indices
        track the generated text.

    Raises:
        TypeError: If the output matches no completion format, including a batch line
            whose body matches none.
        ValidationError: If it matches one but carries an invalid field, named by the
            error pydantic raised.
        ValueError: If a batch line carries no completion, because its request failed.
    """
    try:
        response = LogprobsMixin.parse(outputs)
    except TypeError:
        # A collection of responses is read by iterating it, whatever holds it: a list, a
        # tuple, a 1-D object array, a Series. A single response is tried first because a
        # mapping and a pydantic model iterate too, and neither is a batch.
        if not is_bearable(outputs, Collection) or is_bearable(outputs, str | bytes | Mapping):
            raise
        return [sequence for item in outputs for sequence in parse_top_logprobs(item)]

    return response.top_logprobs()


def parse_sampled_token_logprobs(outputs: Any) -> list[np.ndarray]:
    """Extract the logprob of each *sampled* token, ignoring the alternatives.

    The counterpart to `parse_top_logprobs`, which reads the top-k distribution. Neither
    EPR nor WEPR uses this; it is exposed for callers measuring sequence likelihood.

    Args:
        outputs: A single completion response, as a mapping or an object.

    Returns:
        One 1-D array per sequence, holding the sampled token logprobs in order.

    Raises:
        TypeError: If the output matches no completion format, including a batch line
            whose body matches none.
        ValidationError: If it matches one but carries an invalid field, named by the
            error pydantic raised.
        ValueError: If a batch line carries no completion, because its request failed.
    """
    return LogprobsMixin.parse(outputs).sampled_logprobs()
