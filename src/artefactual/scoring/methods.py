"""Scoring methods for computing confidence scores from log probabilities."""

from collections.abc import Sequence
from enum import StrEnum, auto
from typing import Any, Literal

import numpy as np
from einops import rearrange, reduce
from plum import dispatch, overload
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from artefactual.scoring.logprobs import Logprobs, Scores


class ScoringMethod(StrEnum):
    """Enumeration of supported scoring methods."""

    NAIVE = auto()  # https://cookbook.openai.com/examples/using_logprobs
    MEAN = auto()
    SUPERVISED_ISOTONIC = auto()
    SUPERVISED_SIGMOID = auto()

    @staticmethod
    def supervised() -> set["ScoringMethod"]:
        """Return the set of supervised scoring methods."""
        return {
            ScoringMethod.SUPERVISED_ISOTONIC,
            ScoringMethod.SUPERVISED_SIGMOID,
        }

    @staticmethod
    def unsupervised() -> set["ScoringMethod"]:
        """Return the set of unsupervised scoring methods."""
        return {ScoringMethod.NAIVE, ScoringMethod.MEAN}


@overload
def score_fn(method: Literal[ScoringMethod.NAIVE], logprobs: Logprobs) -> Scores:  # noqa: ARG001
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
def score_fn(method: Literal[ScoringMethod.MEAN], logprobs: Logprobs) -> Scores:  # noqa: ARG001
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


@overload
def score_fn(
    method: Literal[ScoringMethod.SUPERVISED_SIGMOID, ScoringMethod.SUPERVISED_ISOTONIC],
    logprobs: Logprobs,
    ids: Sequence[int],
    labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Supervised scoring methods using calibration.

    Args:
        method: The scoring method (SUPERVISED_SIGMOID or SUPERVISED_ISOTONIC)
        logprobs: Log probabilities array
        ids: Sample IDs
        labels: True labels

    Returns:
        Tuple of (IDs, scores, labels) for the test split

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
    probs = np.exp(logprobs)
    scores = reduce(probs, "n_samples max_len -> n_samples", "mean")
    scores = rearrange(scores, "(n_samples dim) -> n_samples dim", dim=1)
    _, ids_test, scores_train, scores_test, labels_train, labels_test = train_test_split(
        ids, scores, labels, test_size=0.8, random_state=42
    )
    if method is ScoringMethod.SUPERVISED_ISOTONIC:
        calibration_method = "isotonic"
    elif method is ScoringMethod.SUPERVISED_SIGMOID:
        calibration_method = "sigmoid"
    else:
        msg = f"method {method} is not recognized"
        raise ValueError(msg)
    calibrated = CalibratedClassifierCV(method=calibration_method)
    calibrated.fit(scores_train, labels_train)
    scores = calibrated.predict_proba(scores_test)
    return ids_test, scores[:, 0], labels_test


@dispatch
def score_fn(method: ScoringMethod, logprobs: Any, ids: Any | None = None, labels: Sequence[int] | None = None):
    """Generic scoring function that dispatches to the appropriate implementation.

    Args:
        method: The scoring method to use
        logprobs: Log probabilities array
        ids: Optional sample IDs (required for supervised methods)
        labels: Optional true labels (required for supervised methods)

    Raises:
        NotImplementedError: If the method is not implemented
    """
    error_message = f"Scoring method {method} not implemented"
    raise NotImplementedError(error_message)
