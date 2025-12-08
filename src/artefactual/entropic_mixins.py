"""Shared entropic math for transforming log-probabilities.

This module centralizes the logic for turning 3D log-probability tensors into
entropic contributions that can be consumed by transformers or estimators.
"""

from __future__ import annotations

import numpy as np
from beartype import beartype
from beartype.typing import Annotated
from beartype.vale import Is

EXPECTED_NDIM = 3


class EntropicMathMixin:
    """Utilities to convert log-probabilities into entropic contributions."""

    @beartype
    def compute_entropic_contributions(
        self, x: Annotated[object, Is[lambda a: np.ndim(a) == EXPECTED_NDIM]]
    ):
        """Calculate s = -p * log2(p) for 3D inputs.

        Args:
            X: Array-like of shape (N, L, K) containing log-probabilities.

        Returns:
            Tuple of:
            - s_contributions: (N, L, K) entropic contributions.
            - mask: (N, L) boolean mask indicating valid tokens.
        """
        arr = np.array(x)

        # 1. Masking
        pad_val = getattr(self, "padding_value", 0.0)
        is_padding = np.all(arr == pad_val, axis=2)
        mask = ~is_padding

        # 2. Math
        probs = np.exp(arr)
        log2_probs = arr / np.log(2.0)
        s_contributions = -probs * log2_probs

        # 3. Clean padding
        k_val = getattr(self, "K", arr.shape[2])
        mask_expanded = np.repeat(mask[:, :, np.newaxis], k_val, axis=2)
        s_contributions = np.where(mask_expanded, s_contributions, 0.0)

        return s_contributions, mask
