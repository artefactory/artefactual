"""Estimators that support token-level hallucination scoring."""

from __future__ import annotations

import numpy as np
from beartype import beartype
from beartype.typing import Annotated
from beartype.vale import Is
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils.validation import check_is_fitted

EXPECTED_NDIM = 3


class TokenScoringMixin:
    """Mixin that projects linear model weights back to token-level scores."""

    coef_: np.ndarray
    intercept_: np.ndarray
    K: int

    @beartype
    def transform(
        self,
        x_token_contributions: Annotated[object, Is[lambda a: np.ndim(a) == EXPECTED_NDIM]],
    ):
        """Compute token risk scores from 3D entropic contributions."""
        check_is_fitted(self)
        arr = np.array(x_token_contributions)
        if not hasattr(self, "K"):
            message = "Estimator must have attribute 'K'"
            raise AttributeError(message)

        k_val = int(self.K)
        feature_size = int(self.coef_.shape[1])
        if feature_size % k_val != 0:
            message = f"Feature size {feature_size} not divisible by K={k_val}"
            raise ValueError(message)

        m_blocks = feature_size // k_val
        block_coefs = self.coef_[0].reshape(m_blocks, k_val)
        logits_blocks = np.dot(arr, block_coefs.T) + self.intercept_[0]
        probs_blocks = 1 / (1 + np.exp(-logits_blocks))

        component_weights = getattr(self, "component_weights", None)
        weights = np.ones(m_blocks) if component_weights is None else np.array(component_weights)
        if len(weights) != m_blocks:
            message = f"Expected {m_blocks} component weights, got {len(weights)}"
            raise ValueError(message)

        weights_broad = weights.reshape(1, 1, m_blocks)
        weighted_inverse_sum = np.sum(weights_broad / (probs_blocks + 1e-9), axis=2)
        return np.sum(weights) / weighted_inverse_sum


class TokenScoringLogisticRegression(TokenScoringMixin, LogisticRegression):
    """Logistic regression that can emit token scores via transform()."""

    @beartype
    def __init__(
        self,
        k: Annotated[int, Is[lambda i: i > 0]] = 15,
        component_weights=None,
        **kwargs,
    ):
        self.k = k
        self.K = k
        self.component_weights = component_weights
        super().__init__(**kwargs)


class TokenScoringSGD(TokenScoringMixin, SGDClassifier):
    """Linear SGD classifier that can emit token scores via transform()."""

    @beartype
    def __init__(
        self,
        k: Annotated[int, Is[lambda i: i > 0]] = 15,
        component_weights=None,
        **kwargs,
    ):
        self.k = k
        self.K = k
        self.component_weights = component_weights
        super().__init__(**kwargs)
