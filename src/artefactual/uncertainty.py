"""Uncertainty detection using Entropy Production Rate (EPR).

This module provides a detector for measuring uncertainty in language model outputs
by analyzing the entropy of token probability distributions. Higher EPR scores
indicate greater uncertainty or potential hallucination.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

EPSILON = 1e-12


@runtime_checkable
class LogProbValue(Protocol):
    """Protocol for objects that have a logprob attribute (e.g., vLLM Logprob objects)."""

    logprob: float


@runtime_checkable
class CompletionOutput(Protocol):
    """Protocol for completion output objects (e.g., vLLM CompletionOutput)."""

    logprobs: Sequence[dict[str | int, LogProbValue | float]] | None


@runtime_checkable
class RequestOutput(Protocol):
    """Protocol for request output objects (e.g., vLLM RequestOutput)."""

    outputs: Sequence[CompletionOutput]


class UncertaintyDetector:
    """Entropy Production Rate (EPR) detector for measuring model uncertainty.

    This detector computes uncertainty scores based on the entropy of token
    probability distributions from language model outputs. It analyzes the
    top-K log probabilities for each token to assess the model's confidence.

    The EPR score is calculated as the mean of per-token entropy sums across
    the top-K probability distributions. Higher scores indicate:
    - More uncertainty in model predictions
    - Potential hallucination or unreliable outputs
    - Lower model confidence

    Attributes:
        K: Number of top log probabilities to consider per token (default: 15)

    Example:
        >>> detector = UncertaintyDetector(K=15)
        >>> # Assuming you have vLLM outputs
        >>> epr_scores = detector.compute_epr(vllm_outputs)
        >>> # Get per-token scores as well
        >>> seq_scores, token_scores = detector.compute_epr(
        ...     vllm_outputs, return_tokens=True
        ... )
    """

    def __init__(self, k: int = 15) -> None:
        """Initialize the uncertainty detector.

        Args:
            k: Number of top log probabilities to consider per token.
               Must be positive. Default is 15.

        Raises:
            ValueError: If k is not positive
        """
        if k <= 0:
            msg = f"k must be positive, got {k}"
            raise ValueError(msg)
        self.k = k

    @beartype
    def _entropy_contributions(
        self,
        top_logprobs: Sequence[dict[str | int, LogProbValue | float]],
    ) -> NDArray[np.floating]:
        """Compute entropic contributions s_{k,j} = -p log2(p) for top-K probabilities.

        This method calculates the entropy contribution for each token position
        considering the top-K probability distribution. The entropy is computed
        in bits using log base 2.

        Args:
            top_logprobs: Sequence of dictionaries mapping tokens (str or int IDs) to log probabilities.
                         Each dict represents the top-K candidates for a token position.
                         Values can be either float or objects with a 'logprob' attribute.

        Returns:
            Array of shape (num_tokens, K) containing entropy contributions.
            If input is empty, returns array of shape (1, K) filled with zeros.

        Note:
            - Log probabilities are assumed to be in natural log (base e)
            - Probabilities are normalized across the top-K candidates
            - Arrays are padded with zeros if fewer than K candidates exist
            - Arrays are truncated if more than K candidates exist
        """
        if not top_logprobs:
            return np.empty((0, self.k), dtype=np.float32)

        s_kj = []
        for logprob_dict in top_logprobs:
            if not logprob_dict:
                s_kj.append(np.zeros(self.k, dtype=np.float32))
                continue

            logprob_values = [v.logprob if isinstance(v, LogProbValue) else float(v) for v in logprob_dict.values()]

            if not logprob_values:
                s_kj.append(np.zeros(self.k, dtype=np.float32))
                continue

            # Convert to probabilities (logprobs are in natural log, base e)
            logprob_array = np.array(logprob_values, dtype=np.float32)
            probs = np.exp(logprob_array)

            # Normalize top-K probs to sum to 1
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                # Handle edge case of all zeros
                probs = np.ones_like(probs) / len(probs)

            # Calculate entropy contributions in bits (use log2)
            # s = -p * log2(p), with special handling for p=0
            with np.errstate(divide="ignore", invalid="ignore"):
                s = -probs * np.log2(probs + EPSILON)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

            # Pad or truncate to k elements
            if len(s) < self.k:
                s = np.pad(s, (0, self.k - len(s)), mode="constant")
            else:
                s = s[: self.k]

            s_kj.append(s)

        return np.array(s_kj, dtype=np.float32)


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


class WEPR(UncertaintyDetector):
    """Computes Weighted Entropy Production Rate (WEPR)."""

    def compute(self, outputs: Sequence[RequestOutput], weights: list[float], *, return_tokens: bool = False):
        """
        Computes a weighted average of per-token entropy sums.
        The weights could be based on token position, importance, etc.
        """
        # This method would also call self._entropy_contributions()
        # but would apply a weighting scheme before averaging.

        # Example logic:
        # seq_scores = []
        # for out in outputs:
        #     ...
        #     s_kj = self._entropy_contributions(token_logprobs)
        #     token_epr = s_kj.sum(axis=1)
        #
        #     # Apply weights
        #     if len(token_epr) != len(weights):
        #         raise ValueError("Length of weights must match number of tokens.")
        #
        #     weighted_seq_epr = np.average(token_epr, weights=weights)
        #     seq_scores.append(weighted_seq_epr)
        # ...
        pass
