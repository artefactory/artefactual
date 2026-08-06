"""Reduction of entropy contributions into the feature vector a calibration consumes."""

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags

from artefactual.exceptions import EmptySequenceWarning
from artefactual.scoring.entropy_methods.entropy_contributions import EntropyContributionsMixin


def _epr(x, axis) -> np.ndarray:
    """Mean entropy contribution per rank, pooled over the token axis.

    Matches the `mean_entropy` coefficient the calibrations are fit on: the per-rank
    contribution columns averaged, not summed.

    The rank mean divides by `k`, the full rank width, so a NaN rank counts as 0 in the
    numerator and still counts in the denominator; `nanmean` would drop it from both. A
    token whose ranks are entirely NaN is padding rather than data and is excluded from
    the token mean instead.
    """
    is_nan = np.isnan(x)
    padded = np.all(is_nan, axis=-1, keepdims=True)  # fully-NaN (padded) tokens
    s = np.nansum(x, axis=-1, keepdims=True) / x.shape[-1]  # mean over k (rank axis)
    s = np.where(padded, np.nan, s)  # nansum gave 0 for padded tokens → restore NaN
    return np.nanmean(s, axis=axis)  # pool over the token axis


def _wepr(x, axis) -> np.ndarray:
    mean_branch = np.nanmean(x, axis=axis)
    max_branch = np.nanmax(x, axis=axis)
    return np.concatenate([mean_branch, max_branch], axis=-1)


STRATEGIES = {"epr": _epr, "wepr": _wepr}


class EntropyTransformer(BaseEstimator, TransformerMixin, EntropyContributionsMixin):
    """Reduce per-rank entropy contributions to the features a calibration was fit on.

    Input is the `(n_sequences, n_tokens, k)` array `LogProbParser` emits, NaN-padded on
    both the token and rank axes. Output is one feature row per sequence: 1 column for
    `epr`, `2k` for `wepr`.

    Stateless, and takes no rank count -- the width of the input is already the calibrated
    `k`, since the parser sizes the rank axis.

    Args:
        reduction: `"epr"`, `"wepr"`, or a callable `(s_kj, axis) -> features` reducing the
            token axis.
    """

    def __init__(self, reduction: str | Callable = "epr") -> None:
        self.reduction = reduction  # "epr" | "wepr" | custom reduction(s_kj, axis)

    def __sklearn_tags__(self) -> Tags:
        """Declares the transformer stateless and NaN-tolerant."""
        tags = super().__sklearn_tags__()
        tags.requires_fit = False  # stateless: fit learns nothing from data
        tags.input_tags.allow_nan = True  # consumes NaN-padded input on purpose (else check_estimators_nan_inf fails)
        return tags

    def fit(self, _x, _y=None) -> "EntropyTransformer":
        """No-op, present so the step composes in a `Pipeline`. Returns self."""
        return self

    @property
    def reduction_fn(self) -> Callable:
        """The callable named by `reduction`, resolved per call.

        Resolved here rather than in `__init__` because sklearn requires constructor
        arguments to survive `get_params`/`set_params`/`clone` unmodified.

        Raises:
            ValueError: If `reduction` is neither a known name nor callable.
        """
        if callable(self.reduction):
            return self.reduction
        if self.reduction in STRATEGIES:
            return STRATEGIES[self.reduction]
        msg = f"Invalid reduction: {self.reduction!r}. Expected 'epr', 'wepr', or a callable."
        raise ValueError(msg)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Reduce each sequence to one feature row.

        A sequence with no tokens has no entropy to measure. Its features are set to zero,
        which makes the downstream classifier return `sigmoid(intercept_)`, its base rate,
        and a warning is raised rather than letting the degenerate input pass unremarked.

        Args:
            x: `(n_sequences, n_tokens, k)` logprobs, NaN-padded.

        Returns:
            `(n_sequences, n_features)` — 1 column for `epr`, `2k` for `wepr`.

        Warns:
            EmptySequenceWarning: If any sequence carries no tokens.
        """
        s_kj = self.entropy_contributions(x)
        features = self.reduction_fn(s_kj, axis=1)
        empty = np.isnan(features).all(axis=1)  # token-less sequences → all-NaN row
        if empty.any():
            warnings.warn(  # surface the degenerate input, don't pass silently
                f"{int(empty.sum())} empty (token-less) sequence(s); scoring at baseline sigmoid(intercept_).",
                EmptySequenceWarning,
                stacklevel=2,
            )
        features[empty] = 0.0  # zero features → classifier returns baseline
        return features

    def transform_tokens(self, x: np.ndarray) -> np.ndarray:
        """Reduce each token separately, keeping the token axis.

        Applies the same reduction over a one-token window per position, so token features
        are computed exactly as sequence features are.

        Args:
            x: `(n_sequences, n_tokens, k)` logprobs, NaN-padded.

        Returns:
            `(n_sequences, n_tokens, n_features)`, padded tokens still NaN.
        """
        s_kj = self.entropy_contributions(x)
        windows = np.expand_dims(s_kj, axis=2)  # (n, T, 1, k) — one 1-token window per token
        return self.reduction_fn(windows, axis=2)
