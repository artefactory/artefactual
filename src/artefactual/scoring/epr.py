from collections.abc import Sequence

import beartype
import numpy as np
from numpy.typing import NDArray
from vllm import RequestOutput

from artefactual.scoring.uncertainty import UncertaintyDetector


class EPR(UncertaintyDetector):
    """Alias for UncertaintyDetector using Entropy Production Rate (EPR) terminology."""

    @beartype
    def compute(
        self,
        model_responses: Sequence[RequestOutput],
        *,
        return_per_token_scores: bool = False,
    ) -> list[float] | tuple[list[float], list[NDArray[np.floating]]]:
        """Compute EPR-based uncertainty scores for model outputs.

        This method calculates sequence-level EPR scores by:
        1. Computing entropy contributions for each token's top-K probabilities
        2. Summing contributions across K for each token (token-level EPR)
        3. Averaging token-level EPR across the sequence (sequence-level EPR)

        Args:
            outputs: Sequence of model outputs (e.g., vLLM RequestOutput objects).
                    Each output should have a completion with log probabilities.
            return_tokens: If True, also return per-token EPR scores in addition
                          to sequence-level scores. Default is False.

        Returns:
            If return_tokens is False:
                List of sequence-level EPR scores (one float per output)
            If return_tokens is True:
                Tuple of (sequence_scores, token_scores) where:
                - sequence_scores: List of sequence-level EPR scores
                - token_scores: List of arrays with per-token EPR scores

        Raises:
            ValueError: If model_responses is empty
            AttributeError: If model_responses don't have the expected structure

        Example:
            >>> detector = UncertaintyDetector(K=10)
            >>> seq_scores = detector.compute(model_responses)
            >>> print(f"Mean EPR: {np.mean(seq_scores):.3f}")
            Mean EPR: 0.456

            >>> # Get per-token scores
            >>> seq_scores, token_scores = detector.compute(
            ...     model_responses, return_per_token_scores=True
            ... )
            >>> print(f"Tokens in first sequence: {len(token_scores[0])}")
            Tokens in first sequence: 42
        """
        if not model_responses:
            msg = "model_responses cannot be empty"
            raise ValueError(msg)

        seq_scores: list[float] = []
        token_scores: list[NDArray[np.floating]] = []

        for response in model_responses:
            if not response.outputs:
                continue

            for completion in response.outputs:
                token_logprobs = completion.logprobs

                if not token_logprobs:
                    # Handle completions without logprobs
                    seq_scores.append(0.0)
                    if return_per_token_scores:
                        token_scores.append(np.array([], dtype=np.float32))
                    continue

                # Compute entropy contributions matrix (num_tokens, K)
                s_kj = self._entropy_contributions(token_logprobs)

                # Token-level EPR: sum across K for each token
                token_epr = s_kj.sum(axis=1)  # shape: (num_tokens,)

                # Sequence-level EPR: mean across tokens
                if len(token_epr) > 0:
                    seq_epr = float(token_epr.mean())
                else:
                    seq_epr = 0.0

                seq_scores.append(seq_epr)

                if return_per_token_scores:
                    token_scores.append(token_epr)

        if return_per_token_scores:
            return seq_scores, token_scores
        return seq_scores
