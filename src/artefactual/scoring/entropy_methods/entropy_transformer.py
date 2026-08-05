import warnings
from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags

from artefactual.exceptions import EmptySequenceWarning
from artefactual.scoring.entropy_methods.entropy_contributions import EntropyContributionsMixin, RankAlignmentMixin


def _epr(x, axis) -> np.ndarray:
    is_nan = np.isnan(x)
    padded = np.all(is_nan, axis=-1, keepdims=True)  # fully-NaN (padded) tokens
    s = np.nansum(x, axis=-1, keepdims=True)  # sum over k (rank axis)
    s = np.where(padded, np.nan, s)  # nansum gave 0 for padded tokens → restore NaN
    return np.nanmean(s, axis=axis)  # pool over the token axis


def _wepr(x, axis) -> np.ndarray:
    mean_branch = np.nanmean(x, axis=axis)
    max_branch = np.nanmax(x, axis=axis)
    return np.concatenate([mean_branch, max_branch], axis=-1)


STRATEGIES = {"epr": _epr, "wepr": _wepr}


class EntropyTransformer(BaseEstimator, TransformerMixin, EntropyContributionsMixin, RankAlignmentMixin):
    def __init__(self, reduction: str | Callable = "epr", k: int | None = None) -> None:
        self.reduction = reduction  # "epr" | "wepr" | custom reduction(s_kj, axis)
        # Rank count the downstream calibration expects. `None` reduces over whatever
        # width the input carries, which is what a standalone transformer should do; the
        # detector factories always pass an explicit k.
        self.k = k

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.requires_fit = False  # stateless: fit learns nothing from data
        tags.input_tags.allow_nan = True  # consumes NaN-padded input on purpose (else check_estimators_nan_inf fails)
        return tags

    def fit(self, _x, _y=None) -> "EntropyTransformer":
        return self

    @property
    def reduction_fn(self) -> Callable:
        # self.reduction must stay unchanged for clone/get_params/set_params.
        if callable(self.reduction):
            return self.reduction
        if self.reduction in STRATEGIES:
            return STRATEGIES[self.reduction]
        msg = f"Invalid reduction: {self.reduction!r}. Expected 'epr', 'wepr', or a callable."
        raise ValueError(msg)

    def _align(self, s_kj: np.ndarray) -> np.ndarray:
        """Match the rank axis to `self.k`, or leave it alone when no k was given."""
        if self.k is None:
            return s_kj
        return self.align_preserving_padding(s_kj, self.k)

    def transform(self, x: np.ndarray) -> np.ndarray:  # (n, n_features)
        s_kj = self._align(self.entropy_contributions(x))
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

    def transform_tokens(self, x: np.ndarray) -> np.ndarray:  # (n, T, n_features)
        s_kj = self._align(self.entropy_contributions(x))
        windows = np.expand_dims(s_kj, axis=2)  # (n, T, 1, k) — one 1-token window per token
        return self.reduction_fn(windows, axis=2)
