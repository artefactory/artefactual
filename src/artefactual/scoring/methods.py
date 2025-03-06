"""Scoring methods for computing confidence scores from log probabilities.

This module provides various methods for converting log probabilities into
confidence scores, including both unsupervised and supervised approaches.
"""

from collections.abc import Sequence
from enum import StrEnum, auto
from typing import Literal

import numpy as np
from einops import rearrange, reduce
from plum import dispatch, overload
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Use the types from logprobs.py
from artefactual.scoring.logprobs import LogProbs, Scores


class ScoringMethod(StrEnum):
    """Enumeration of supported scoring methods.

    This enum defines the available methods for converting log probabilities
    into confidence scores.

    Attributes:
        NAIVE: Simple method that uses the sum of log probabilities
        MEAN: Uses the mean of log probabilities
        SUPERVISED_ISOTONIC: Supervised method using isotonic regression
        SUPERVISED_SIGMOID: Supervised method using logistic regression
    """

    NAIVE = auto()  # https://cookbook.openai.com/examples/using_logprobs
    MEAN = auto()
    SUPERVISED_ISOTONIC = auto()
    SUPERVISED_SIGMOID = auto()


@overload
def score_fn(method: Literal[ScoringMethod.NAIVE], logprobs: LogProbs) -> Scores:  # noqa: ARG001
    """Naive scoring method that uses sum of log probabilities.

    Args:
        method: The scoring method (NAIVE)
        logprobs: Log probabilities array

    Returns:
        Confidence scores

    Example:
        >>> import numpy as np
        >>> logprobs = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
        >>> score_fn(ScoringMethod.NAIVE, logprobs)
        array([0.0025], dtype=float32)
    """
    return np.exp(reduce(logprobs, "batch max_len -> batch", "sum"))


@overload
def score_fn(method: Literal[ScoringMethod.MEAN], logprobs: LogProbs) -> Scores:  # noqa: ARG001
    """Mean scoring method that uses mean of log probabilities.

    Args:
        method: The scoring method (MEAN)
        logprobs: Log probabilities array

    Returns:
        Confidence scores

    Example:
        >>> import numpy as np
        >>> logprobs = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
        >>> score_fn(ScoringMethod.MEAN, logprobs)
        array([0.1353], dtype=float32)
    """
    return np.exp(reduce(logprobs, "batch max_len -> batch", "mean"))


MIN_SAMPLES_PER_CLASS = 2
N_CV = 3
TEST_SIZE = 0.8


@overload
def score_fn(
    method: Literal[ScoringMethod.SUPERVISED_SIGMOID, ScoringMethod.SUPERVISED_ISOTONIC],
    logprobs: LogProbs,
    ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Supervised scoring methods using calibration.

    Args:
        method: The scoring method (SUPERVISED_SIGMOID or SUPERVISED_ISOTONIC)
        logprobs: Log probabilities array
        ids: Sample IDs
        labels: True labels

    Returns:
        Tuple of (test IDs, calibrated scores, test labels)

    Raises:
        ValueError: If the method is not a supervised method
        ValueError: If ids or labels have different lengths from logprobs

    Example:
        >>> import numpy as np
        >>> logprobs = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
        >>> ids = np.array([1, 2])
        >>> labels = np.array([1, 0])
        >>> test_ids, scores, test_labels = score_fn(
        ...     ScoringMethod.SUPERVISED_SIGMOID, logprobs, ids, labels
        ... )
        >>> len(scores) > 0
        True
    """

    if len(ids) != len(labels) or len(ids) != len(logprobs):
        error_msg = f"Length mismatch: ids={len(ids)}, labels={len(labels)}, logprobs={len(logprobs)}"
        raise ValueError(error_msg)

    # Check for minimum required samples
    min_required_samples = MIN_SAMPLES_PER_CLASS * N_CV * 2
    if len(logprobs) < min_required_samples:
        error_msg = f"Insufficient samples: {len(logprobs)} provided, but {min_required_samples} required."
        raise ValueError(error_msg)

    # Convert to probabilities
    # Use naive scoring method (sum of logprobs, then exp)
    scores = np.exp(reduce(logprobs, "n_samples max_len -> n_samples", "sum"))
    scores = rearrange(scores, "n_samples -> n_samples 1")

    # Split into train and test sets
    _, ids_test, scores_train, scores_test, labels_train, labels_test = train_test_split(
        ids, scores, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    # Select calibration method
    if method is ScoringMethod.SUPERVISED_ISOTONIC:
        calibration_method = "isotonic"
    elif method is ScoringMethod.SUPERVISED_SIGMOID:
        calibration_method = "sigmoid"
    else:
        error_msg = f"Method {method} is not a supported supervised method"
        raise ValueError(error_msg)

    # Train calibration model with cross-validation if we have enough samples
    calibrated: BaseEstimator = CalibratedClassifierCV(method=calibration_method, cv=N_CV)
    calibrated.fit(scores_train, labels_train)

    # Get calibrated probabilities
    calibrated_scores = calibrated.predict_proba(scores_test)

    return ids_test, calibrated_scores[:, 1], labels_test


@dispatch
def score_fn(
    method: ScoringMethod,
    logprobs: LogProbs,
    ids: Sequence[int] | None = None,
    labels: Sequence[int] | None = None,
) -> Scores | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic scoring function that dispatches to the appropriate implementation.

    This function routes to the specific implementation based on the scoring method.
    For supervised methods, ids and labels must be provided.

    Args:
        method: The scoring method to use
        logprobs: Log probabilities array
        ids: Optional sample IDs (required for supervised methods)
        labels: Optional true labels (required for supervised methods)

    Returns:
        Either scores array (for unsupervised methods) or
        tuple of (test IDs, calibrated scores, test labels) for supervised methods
    """
    pass  # type: ignore
