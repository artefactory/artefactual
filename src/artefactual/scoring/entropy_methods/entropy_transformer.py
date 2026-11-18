"""Reduction of entropy contributions into the feature vector a calibration consumes."""

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags
from sklearn.utils.validation import _check_feature_names_in, check_array, validate_data

from artefactual.exceptions import EmptySequenceWarning
from artefactual.scoring.entropy_methods.entropy_contributions import EntropyContributionsMixin


def _nan_reduce(reduction: Callable, x: np.ndarray, axis: int) -> np.ndarray:
    """Apply a NaN-aware numpy reduction, silencing its all-NaN-slice warnings.

    A fully padded sequence — or a padded token position, in `transform_tokens` — is an
    expected input here, not an anomaly: `transform` detects it and raises
    `EmptySequenceWarning`, which names the condition and says what the score will be.
    numpy's "Mean of empty slice" and "All-NaN slice encountered" report the same thing a
    second time, from a frame that points at this file rather than at the caller's data.

    Scoped to these reductions rather than to `transform`, so a caller-supplied `reduction`
    keeps whatever warnings it raises.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        return reduction(x, axis=axis)


def _epr(x, axis) -> np.ndarray:
    """Entropy Production Rate: the truncated entropy, averaged over tokens.

    The paper's Eq. 6 -- the mean over the sequence of the top-K entropy estimate
    `H~_K = -sum_k p_k ln p_k` (Eq. 3). So the rank axis is summed, not averaged: the
    result is an entropy, on the same scale as `H(q)` itself, in nats as the ECIR2026
    release computes it.

    A token whose candidates are entirely NaN is padding rather than data, and is excluded from
    the token mean rather than contributing the 0 that `nansum` would give it.
    """
    padded = np.all(np.isnan(x), axis=-1, keepdims=True)  # fully-NaN (padded) tokens
    s = np.nansum(x, axis=-1, keepdims=True)  # H~_K: sum over the K ranks
    s = np.where(padded, np.nan, s)  # nansum gave 0 for padded tokens → restore NaN
    return _nan_reduce(np.nanmean, s, axis)  # EPR: mean over the token axis


def _wepr(x, axis) -> np.ndarray:
    mean_branch = _nan_reduce(np.nanmean, x, axis)
    max_branch = _nan_reduce(np.nanmax, x, axis)
    return np.concatenate([mean_branch, max_branch], axis=-1)


STRATEGIES = {"epr": _epr, "wepr": _wepr}


# Base order is significant. `__sklearn_tags__` resolves through `super()`, so
# `TransformerMixin` must precede `BaseEstimator` for `transformer_tags` to be set.
class EntropyTransformer(TransformerMixin, BaseEstimator, EntropyContributionsMixin):
    """Reduce per-candidate entropy contributions to the features a calibration was fit on.

    Input is the `(n_sequences, n_tokens, k)` array `LogProbParser` emits, NaN-padded on
    both the token and rank axes, or a 2-D `(n_sequences, width)` table of per-rank
    logprobs, which is read as one token per row. Output is one feature row per sequence:
    1 column for `epr`, `2k` for `wepr`.

    Learns nothing, but `fit` records the width and column names of a 2-D table, as
    scikit-learn's validation does, so the features of a named table are named after its
    columns. `k` is optional: the width of a parsed cube is already the calibrated `k`,
    since the parser sizes the rank axis. Giving `k` keeps only the first `k` ranks of
    wider input, which are the `k` most likely candidates, and names the `wepr` features
    without being shown any data, which is what scikit-learn's `set_output` needs to label
    the columns of `transform`. Narrower input is used as it is.

    Parameters
    ----------
    reduction : {"epr", "wepr"} or callable, default="epr"
        A name, or a callable `(s_kj, axis) -> features` reducing the token axis.
    k : int, default=None
        Candidates per token to keep, the first `k` ranks. `None` keeps every rank, and then a `wepr` step
        cannot name its features without `input_features` or a fitted table.

    Examples
    --------
    >>> features = EntropyTransformer("wepr").transform(logprobs)  # doctest: +SKIP
    >>> features.shape  # (n_sequences, 2 * k)  # doctest: +SKIP
    """

    def __init__(self, reduction: str | Callable = "epr", k: int | None = None) -> None:
        self.reduction = reduction  # "epr" | "wepr" | custom reduction(s_kj, axis)
        self.k = k

    def __sklearn_tags__(self) -> Tags:
        """Declares the transformer stateless and NaN-tolerant."""
        tags = super().__sklearn_tags__()
        tags.requires_fit = False  # stateless: fit learns nothing from data
        tags.input_tags.allow_nan = True  # consumes NaN-padded input on purpose (else check_estimators_nan_inf fails)
        # Input is three-dimensional: `transform` reduces the token axis and the reduction
        # reduces the rank axis, and the two coincide below three dimensions. The tags select
        # which arrays scikit-learn's checks synthesise, and the only producer of this step's
        # input, `LogProbParser`, emits `(n_sequences, n_tokens, k)`.
        tags.input_tags.two_d_array = False
        tags.input_tags.three_d_array = True
        return tags

    def validate(self, X, *, reset: bool = False) -> np.ndarray:  # noqa: N803
        """`X` as a float `(n, tokens, width)` array, cut to the first `k` ranks.

        A 3-D input is the parser's cube: NaN is the padding marker and passes, `inf` is no
        logprob and would reduce to a meaningless feature, so it is refused. Anything else
        is validated as a numeric 2-D table, which records (`reset=True`) or checks
        (`reset=False`) `n_features_in_` and the column names, and is read as one token
        per row. A list converts, so the step takes what `np.asarray` would. Empty batches
        pass: the parser returns an empty cube for them.

        Parameters
        ----------
        X : array-like
            A 3-D cube or a 2-D table of per-rank logprobs.
        reset : bool, default=False
            Record the width and column names of a 2-D table, as `fit` does, instead of
            checking them against the recorded ones. Ignored for a cube.

        Returns
        -------
        ndarray of shape (n_sequences, n_tokens, width)
            `width` is `k` when `k` is set and the input is at least that wide.

        Raises
        ------
        ValueError
            If a cube holds infinity, or a table is not numeric 2-D or disagrees with the
            fitted width or column names.
        """
        if np.asarray(X).ndim == 3:  # not np.ndim: an array-like that only defines `__array__` is refused by it
            cube = check_array(
                X,
                ensure_2d=False,
                allow_nd=True,
                ensure_all_finite="allow-nan",
                dtype=[np.float64, np.float32],
                ensure_min_samples=0,
                ensure_min_features=0,
            )
        else:
            cube = validate_data(self, X, reset=reset, dtype=np.float64)[:, None, :]
        return cube if self.k is None else cube[..., : self.k]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the features `transform` returns.

        Parameters
        ----------
        input_features : array-like of str, default=None
            Names of the rank columns the step reads, as the parser names them
            (`rank0`, ...); its length must equal `k` when `k` is set, and it must equal
            the column names the step was fitted on, if any. Without it the names are
            the fitted table's columns (the first `k`), or come from `k`.

        Returns
        -------
        ndarray of str
            `["epr"]`, or for `wepr` `wepr_mean_<rank>` for each rank followed by
            `wepr_max_<rank>` for each rank. A rank is named by the column it was read
            from when the step was fitted on a named table.

        Raises
        ------
        ValueError
            If `reduction` is a callable; if `input_features` does not have `k` entries or
            disagrees with the fitted column names; or if `reduction` is `"wepr"` and none
            of `k`, `input_features` or fitted column names is available.
        """
        if callable(self.reduction):
            msg = "Feature names are unknown for a custom reduction callable; only 'epr' and 'wepr' are named."
            raise ValueError(msg)  # noqa: TRY004
        if input_features is not None and self.k is not None and len(input_features) != self.k:
            msg = f"input_features has {len(input_features)} entries but the step reads k={self.k} candidates."
            raise ValueError(msg)
        if self.reduction_fn is _epr:
            return np.array(["epr"], dtype=object)
        names = _check_feature_names_in(self, input_features, generate_names=False)
        if names is not None:
            ranks = [str(name) for name in names[: self.k]]
        elif self.k is not None:
            ranks = [f"rank{i}" for i in range(self.k)]
        else:
            msg = "A wepr step with k=None holds no width: set k or pass input_features to name its features."
            raise ValueError(msg)
        # `_wepr` concatenates the mean branch, then the max branch, on the last axis.
        return np.array([f"wepr_{branch}_{rank}" for branch in ("mean", "max") for rank in ranks], dtype=object)

    def fit(self, X, y=None) -> "EntropyTransformer":  # noqa: ARG002, N803 — y unused but required by the sklearn fit signature
        """Learns nothing; a 2-D table is validated so its width and column names are recorded.

        Parameters
        ----------
        X : array-like, default=None
            A 3-D cube is only validated; a 2-D table sets `n_features_in_` and
            `feature_names_in_`. `None` is accepted, and fits nothing.
        y : array-like, default=None
            Ignored.

        Returns
        -------
        self : EntropyTransformer
        """
        if X is not None:
            self.validate(X, reset=True)
        return self

    @property
    def reduction_fn(self) -> Callable:
        """The callable named by `reduction`, resolved per call.

        Resolved here rather than in `__init__` because sklearn requires constructor
        arguments to survive `get_params`/`set_params`/`clone` unmodified.

        Raises
        ------
        ValueError
            If `reduction` is neither a known name nor callable.
        """
        if callable(self.reduction):
            return self.reduction
        if self.reduction in STRATEGIES:
            return STRATEGIES[self.reduction]
        msg = f"Invalid reduction: {self.reduction!r}. Expected 'epr', 'wepr', or a callable."
        raise ValueError(msg)

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Reduce each sequence to one feature row.

        A sequence with no tokens has no entropy to measure. Its features are set to zero,
        which makes the downstream classifier return `sigmoid(intercept_)`, its base rate,
        and a warning is raised rather than letting the degenerate input pass unremarked.

        Parameters
        ----------
        X : array-like of shape (n_sequences, n_tokens, k) or (n_sequences, k)
            Logprobs, NaN-padded; a nested list is accepted. A 2-D table is one token per
            row. Ranks beyond `k` are dropped.

        Returns
        -------
        ndarray of shape (n_sequences, n_features)
            1 column for `epr`, `2k` for `wepr`. After `set_output(transform="pandas")` or
            `"polars"`, a frame whose columns are `get_feature_names_out()`.

        Raises
        ------
        ValueError
            If `X` holds infinity, or is a table that is not numeric 2-D.

        Warns
        -----
        EmptySequenceWarning
            If any sequence carries no tokens.
        """
        s_kj = self.entropy_contributions(self.validate(X))
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

    def transform_tokens(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Reduce each token separately, keeping the token axis.

        Applies the same reduction over a one-token window per position, so token features
        are computed exactly as sequence features are.

        Parameters
        ----------
        X : ndarray of shape (n_sequences, n_tokens, k) or (n_sequences, k)
            Logprobs, NaN-padded. A 2-D table is one token per row. Ranks beyond `k` are
            dropped.

        Returns
        -------
        ndarray of shape (n_sequences, n_tokens, n_features)
            Padded tokens are still NaN. Never wrapped by `set_output`.

        Raises
        ------
        ValueError
            If `X` holds infinity, or is a table that is not numeric 2-D.
        """
        s_kj = self.entropy_contributions(self.validate(X))
        windows = np.expand_dims(s_kj, axis=2)  # (n, T, 1, k) — one 1-token window per token
        return self.reduction_fn(windows, axis=2)
