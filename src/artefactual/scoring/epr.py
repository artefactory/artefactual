from collections.abc import Sequence

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from artefactual.data.data_model import Completion
from artefactual.scoring.uncertainty_detector import UncertaintyDetector


class EPR(UncertaintyDetector):
    """Computes Entropy Production Rate (EPR) from model completions."""

    @beartype
    def compute(
        self,
        completions: Sequence[Completion],
        *,
        return_per_token_scores: bool = False,
    ) -> list[float] | tuple[list[float], list[NDArray[np.floating]]]:
        """
        Compute EPR-based uncertainty scores from a sequence of completions.
        Args:
            completions: A list of completions, where each completion is a list of tokens,
                         and each token is a list of its top-K log probabilities.
                         Shape: (num_completions, num_tokens, num_logprobs)
        Returns:
            - List of sequence-level EPR scores.
            - Optionally (if return_per_token_scores=True),
              a tuple of (seq_scores, per_token_scores).
        """
        if not completions:
            return []

        seq_scores: list[float] = []
        token_scores: list[NDArray[np.floating]] = []

        for completion in completions:
            token_logprobs_dict = completion.token_logprobs
            if not token_logprobs_dict:
                seq_scores.append(0.0)
                if return_per_token_scores:
                    token_scores.append(np.array([], dtype=np.float32))
                continue

            # Convert to a 2D numpy array for vectorized processing
            # Sort by token position to ensure correct order
            sorted_indices = sorted(token_logprobs_dict.keys())
            logprobs_list = [token_logprobs_dict[i] for i in sorted_indices]

            # Compute entropy contributions in a vectorized manner
            # Input shape: (num_tokens, K)
            s_kj = self._entropy_contributions(logprobs_list)

            # Token-level EPR: sum across K
            token_epr = s_kj.sum(axis=1)  # shape (num_tokens,)

            # Sequence-level EPR: mean of tokens
            seq_epr = float(token_epr.mean()) if token_epr.size > 0 else 0.0
            seq_scores.append(seq_epr)

            if return_per_token_scores:
                token_scores.append(token_epr)

        if return_per_token_scores:
            return seq_scores, token_scores
        return seq_scores
