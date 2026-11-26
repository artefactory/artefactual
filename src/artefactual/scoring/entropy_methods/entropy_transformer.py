"""Pooling of entropy contributions into the feature vector a calibration consumes."""

import warnings
from collections.abc import Callable
from copy import copy
from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.exceptions import NotFittedError
from sklearn.utils import Tags
from sklearn.utils.validation import _check_feature_names_in, check_array, check_is_fitted, validate_data

from artefactual.exceptions import EmptySequenceWarning
from artefactual.scoring.entropy_methods.entropy_contributions import EntropyContributionsMixin
from artefactual.scoring.entropy_methods.pooling import Pooling, as_pooling, epr_reduction, wepr_reduction

# The reductions live in `pooling`; they stay importable from here.
__all__ = ["EntropyTransformer", "epr_reduction", "wepr_reduction"]


def fitted_width(step: Any) -> int | None:  # Any: `n_features_in_` is set by fit, so no estimator type declares it
    """The width `step` was fitted on, or `None` before it was fitted."""
    try:
        check_is_fitted(step, "n_features_in_")
    except NotFittedError:
        return None
    return step.n_features_in_


# Base order is significant. `__sklearn_tags__` resolves through `super()`, so
# `TransformerMixin` must precede `BaseEstimator` for `transformer_tags` to be set.
class EntropyTransformer(TransformerMixin, BaseEstimator, EntropyContributionsMixin):
    """Reduce per-candidate entropy contributions to the features a calibration was fit on.

    Input is the `(n_sequences, n_tokens, width)` array `LogProbParser` emits, NaN-padded on
    both the token and rank axes, or a 2-D `(n_sequences, width)` table of per-rank
    logprobs, which is read as one token per row. Output is one feature row per sequence,
    as wide as the pooling makes it: 1 column for `EPRPooling`, `2 * width` for `WEPRPooling`.

    Learns nothing, but `fit` records the width of the input and the column names of a 2-D
    table, as scikit-learn's validation does, so the features of a named table are named
    after its columns. A fitted step refuses input of another width: the estimators after it
    were fitted on the features of that width.

    Parameters
    ----------
    pooling : Pooling or callable
        Required: no pooling is chosen for you. A `Pooling` instance (`EPRPooling()`,
        `WEPRPooling()`, or a subclass of your own), or a callable `f(x, axis) -> (..., n_features)`
        pooling the token `axis` of the NaN-padded contributions `x`, whose last axis is the
        rank axis. Stored as given and wrapped when used, so a string raises `TypeError` from `transform` and
        `get_feature_names_out`.

    Examples
    --------
    >>> features = EntropyTransformer(WEPRPooling()).transform(
    ...     logprobs
    ... )  # doctest: +SKIP
    >>> features.shape  # (n_sequences, 2 * width)  # doctest: +SKIP
    >>> EntropyTransformer(pooling=my_pooling).transform(logprobs)  # doctest: +SKIP
    """

    _parameter_constraints: ClassVar[dict] = {
        "pooling": [Pooling, callable],
    }

    def __init__(self, pooling: Pooling | Callable) -> None:
        self.pooling = pooling  # stored untouched: clone checks constructor params by identity; wrapped at use

    def applied_pooling(self) -> Pooling:
        """The `Pooling` that `transform` applies: `pooling` itself, or a callable wrapped.

        Raises
        ------
        TypeError
            If `pooling` is neither a `Pooling` nor callable.
        """
        return as_pooling(self.pooling)

    def __sklearn_tags__(self) -> Tags:
        """Declares the transformer stateless and NaN-tolerant."""
        tags = super().__sklearn_tags__()
        tags.requires_fit = False  # stateless: fit learns nothing from data
        # The tags describe what scikit-learn's checks may synthesise, so they declare both
        # the numeric table the checks can fit and score and the cube `LogProbParser` emits.
        # A table refuses NaN (no rank is "padding" in it), whereas a cube takes NaN as
        # padding inside `validate` whatever `allow_nan` says.
        tags.input_tags.allow_nan = False
        tags.input_tags.two_d_array = True
        tags.input_tags.three_d_array = True
        return tags

    def validate(self, X, *, reset: bool = False) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """`X` as a float `(n, tokens, width)` array.

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
            Record the width (and, for a table, the column names), as `fit` does, instead of
            checking them against the recorded ones.

        Returns
        -------
        ndarray of shape (n_sequences, n_tokens, width)
            `width` is the input's last dimension.

        Raises
        ------
        ValueError
            If a cube holds infinity, or a table is not numeric 2-D or disagrees with the
            fitted width or column names, or a cube's last dimension differs from the fitted width.
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
            if reset:
                self.n_features_in_ = cube.shape[-1]  # the width a table records, so names read one attribute
                vars(self).pop("feature_names_in_", None)  # a cube has no column names: drop an earlier table's
            elif cube.size and (fitted := fitted_width(self)) is not None and cube.shape[-1] != fitted:
                msg = f"X has {cube.shape[-1]} features, but {type(self).__name__} is expecting {fitted} features."
                raise ValueError(msg)
        else:
            cube = validate_data(self, X, reset=reset, dtype=np.float64)[:, None, :]
        return cube

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the features `transform` returns.

        Parameters
        ----------
        input_features : array-like of str, default=None
            Names of the rank columns, as the parser names them (`rank0`, ...): one per
            column the step was fitted on. They must equal the column names the step was
            fitted on, if any. Without it the names are the fitted table's columns, or
            `rank0`, ... for the width the step was fitted on.

        Returns
        -------
        ndarray of str
            What the pooling's `feature_names` gives: `["epr"]` for `EPRPooling`;
            `wepr_mean_<rank>` for each rank followed by `wepr_max_<rank>` for each rank for
            `WEPRPooling`; by default the pooling's name for one feature and `name_0`,
            `name_1`, ... for more. A rank is named by the column it was read from when the
            step was fitted on a named table.

        Raises
        ------
        ValueError
            If `input_features` does not have one entry per column fitted on or disagrees
            with the fitted column names; if the pooling's width depends on the number of
            ranks and neither `input_features`, a fit nor fitted column names give it; or
            if its feature names do not match its width.
        TypeError
            If `pooling` is neither a `Pooling` nor callable.
        """
        pooling = self.applied_pooling()
        fitted = fitted_width(self)
        names = _check_feature_names_in(self, input_features, generate_names=False)
        if names is not None:
            ranks = [str(name) for name in names]
        elif fitted is not None:
            ranks = [f"rank{i}" for i in range(fitted)]
        else:
            ranks = []
        if not ranks:
            # Nothing says how many ranks there are: only a pooling that does not depend on
            # the width can name its features, and a placeholder rank stands for the one it is given.
            if not pooling.constant_width:
                msg = (
                    f"Pooling {pooling.name!r} has a width that depends on the number of ranks, which is "
                    "unknown: fit the step or pass input_features to name its features."
                )
                raise ValueError(msg)
            ranks = ["rank0"]
        features = [str(name) for name in pooling.feature_names(ranks)]
        width = pooling.width(len(ranks))
        if len(features) != width:
            msg = (
                f"Pooling {pooling.name!r} returns {width} feature(s) for {len(ranks)} rank(s), "
                f"but its feature_names gave {len(features)}: {features}."
            )
            raise ValueError(msg)
        return np.array(features, dtype=object)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None) -> "EntropyTransformer":  # ruff: ignore[unused-method-argument, invalid-argument-name] — y unused but required by the sklearn fit signature
        """Learns nothing; the input is validated so its width and column names are recorded.

        Parameters
        ----------
        X : array-like or None
            A 3-D cube sets `n_features_in_`, its last dimension; a 2-D table sets it too,
            and `feature_names_in_` when its columns are named. `None` fits nothing.
        y : array-like, default=None
            Ignored.

        Returns
        -------
        self : EntropyTransformer
        """
        self.pooling_ = copy(self.applied_pooling())  # shallow: a wrapped callable stays the same object
        if X is not None:
            self.validate(X, reset=True)
        return self

    def check_pooling_is_the_fitted_one(self) -> Pooling:
        """The `Pooling` to apply, once it is the one the step was fitted with.

        A downstream estimator is fitted on the features of the pooling in force at `fit`, and
        `set_params` does not refit, so a swapped pooling would feed it features it was not
        fitted on. A step that was never fitted has nothing to be held to.

        Raises
        ------
        ValueError
            If `pooling` differs from the one `fit` saw: refit.
        """
        pooling = self.applied_pooling()
        try:
            check_is_fitted(self, "pooling_")
        except NotFittedError:
            return pooling
        if pooling != self.pooling_:
            msg = (
                f"EntropyTransformer was fitted with pooling={self.pooling_!r} but now has pooling={pooling!r}. "
                "The estimators after it were fitted on the features of the former, so refit before transforming."
            )
            raise ValueError(msg)
        return pooling

    def transform(self, X: np.ndarray) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Reduce each sequence to one feature row.

        A sequence with no tokens has no entropy to measure. Its features are set to zero,
        which makes the downstream classifier return `sigmoid(intercept_)`, its base rate,
        and a warning is raised rather than letting the degenerate input pass unremarked.

        Parameters
        ----------
        X : array-like of shape (n_sequences, n_tokens, width) or (n_sequences, width)
            Logprobs, NaN-padded; a nested list is accepted. A 2-D table is one token per
            row.

        Returns
        -------
        ndarray of shape (n_sequences, n_features)
            1 column for `EPRPooling`, `2 * width` for `WEPRPooling`, the pooling's own width otherwise. After
            `set_output(transform="pandas")` or `"polars"`, a frame whose columns are `get_feature_names_out()`.

        Raises
        ------
        ValueError
            If `X` holds infinity, or is a table that is not numeric 2-D.
        TypeError
            If `pooling` is neither a `Pooling` nor callable.

        Warns
        -----
        EmptySequenceWarning
            If any sequence carries no tokens.
        """
        s_kj = self.entropy_contributions(self.validate(X))
        features = self.check_pooling_is_the_fitted_one()(s_kj, 1)
        empty = np.isnan(features).all(axis=1)  # token-less sequences → all-NaN row
        if empty.any():
            warnings.warn(  # surface the degenerate input, don't pass silently
                f"{int(empty.sum())} empty (token-less) sequence(s); scoring at baseline sigmoid(intercept_).",
                EmptySequenceWarning,
                stacklevel=2,
            )
        features[empty] = 0.0  # zero features → classifier returns baseline
        return features

    def transform_tokens(self, X: np.ndarray) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Reduce each token separately, keeping the token axis.

        Applies the same pooling over a one-token window per position, so token features
        are computed exactly as sequence features are. A padding token (all values NaN) is
        not passed to the pooling, so a pooling never sees a window the caller did not write.

        Parameters
        ----------
        X : ndarray of shape (n_sequences, n_tokens, width) or (n_sequences, width)
            Logprobs, NaN-padded. A 2-D table is one token per row.

        Returns
        -------
        ndarray of shape (n_sequences, n_tokens, n_features)
            Padded tokens are NaN rows. Never wrapped by `set_output`.

        Raises
        ------
        ValueError
            If `X` holds infinity, or is a table that is not numeric 2-D.
        """
        s_kj = self.entropy_contributions(self.validate(X))
        pooling = self.check_pooling_is_the_fitted_one()
        real = ~np.isnan(s_kj).all(axis=-1)  # (n, T): False at padding tokens
        windows = s_kj[real][:, None, :]  # (m, 1, width) — one 1-token window per real token
        pooled = np.asarray(pooling(windows, 1))
        out = np.full((*real.shape, pooled.shape[-1]), np.nan, dtype=pooled.dtype)
        out[real] = pooled
        return out
