"""Feature transformers for entropic hallucination scoring."""

from __future__ import annotations

import numpy as np
from beartype import beartype
from beartype.typing import Annotated
from beartype.vale import Is
from sklearn.base import BaseEstimator, TransformerMixin

from artefactual.entropic_mixins import EntropicMathMixin

EXPECTED_NDIM = 3
VALID_STATS = {"mean", "max", "sum", "min"}

class EntropicContributionTransformer(BaseEstimator, TransformerMixin, EntropicMathMixin):
    """Transform 3D log-probs (N, L, K) into entropic contributions (N, L, K)."""

    def __init__(self, k: int = 15, padding_value: float = 0.0):
        self.k = k
        self.K = k
        self.padding_value = padding_value

    def fit(self, _x, _y=None):
        return self

    @beartype
    def transform(self, x: Annotated[object, Is[lambda a: np.ndim(a) == EXPECTED_NDIM]]):
        s, _mask = self.compute_entropic_contributions(x)
        return s

    def get_feature_names_out(self, _input_features=None):
        return np.array([f"entropy_rank_{k}" for k in range(self.K)])


class SequenceStatAggregator(BaseEstimator, TransformerMixin):
    """Aggregate 3D entropic contributions into 2D features per sequence."""

    @beartype
    def __init__(
        self,
        stat: Annotated[str, Is[lambda s: s in VALID_STATS]] = "mean",
        k: Annotated[int, Is[lambda i: i > 0]] = 15,
    ):
        self.stat = stat
        self.k = k
        self.K = k

    def fit(self, _x, _y=None):
        return self

    @beartype
    def transform(self, seq_values: Annotated[object, Is[lambda a: np.ndim(a) == EXPECTED_NDIM]]):
        """Aggregate along the sequence dimension."""
        values = np.array(seq_values)
        is_valid = np.any(values != 0.0, axis=2)
        seq_lengths = np.maximum(np.sum(is_valid, axis=1, keepdims=True), 1.0)

        if self.stat == "mean":
            return np.sum(values, axis=1) / seq_lengths
        if self.stat == "max":
            return np.max(values, axis=1)
        if self.stat == "sum":
            return np.sum(values, axis=1)
        if self.stat == "min":
            filled = values.copy()
            mask_expanded = np.repeat(is_valid[:, :, np.newaxis], self.K, axis=2)
            filled[~mask_expanded] = np.inf
            val = np.min(filled, axis=1)
            val[np.isinf(val)] = 0.0
            return val

        message = f"Unknown stat: {self.stat}"
        raise ValueError(message)

    def get_feature_names_out(self, _input_features=None):
        return np.array([f"{self.stat}_rank_{k}" for k in range(self.K)])
