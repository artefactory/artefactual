"""
Calibration training for epr and wepr detectors.

"""

import logging
import numpy as np
from artefactual.scoring.base_detector import BaseDetector, epr, wepr


logger = logging.getLogger(__name__)

MIN_CLASSES_FOR_TRAINING = 2


def train_calibration(X: list, y: np.ndarray, reduction: str)  -> BaseDetector:
    """
    Fits an EPR or WEPR pipeline on raw JSON responses.

    Args:
        X: Raw json response.
        y: Binary labels - 1 if hallucination, 0 if correct.
        reduction: scoring variants - "epr" or "wepr"

    Returns:
        The fitted BaseDetector pipeline.
    """
    if reduction not in ("epr", "wepr"):
        msg = f"Invalid reduction: {reduction!r}. Expected 'epr' or 'wepr'."
        raise ValueError(msg)

    if len(X) != len(y):
        msg = f"Length of X ({len(X)}) must match length of y ({len(y)})."
        raise ValueError(msg)

    if len(np.unique(y)) < MIN_CLASSES_FOR_TRAINING:
        msg = "Need both positive and negative labels to train."
        raise ValueError(msg)

    logger.info(f"Training on {len(y)} samples.")

    # Fitting BaseDetector
    if reduction == "epr":
        clf = epr()
    else:
        clf = wepr()

    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    logger.info(f"Hallucination probabilities: {scores}")
    return clf
