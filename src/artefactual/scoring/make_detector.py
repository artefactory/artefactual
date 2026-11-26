"""The out-of-the-box detector pipeline."""

from typing import Annotated

import numpy as np
from beartype import beartype
from beartype.vale import Is
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_classifier import WEPR
from artefactual.scoring.tokens import (
    TokenClassifier,
    has_token_layout,
    token_proba,
)
from artefactual.utils.typing import TYPE_CHECKED

TokenDetector = Annotated[TokenClassifier, Is[is_classifier]]


class DetectorPipeline(Pipeline):
    """A scikit-learn `Pipeline` from completion responses to scores, with per-token scores.

    Behaves as a `Pipeline` that has `predict_token_proba`. Its layout is one of: a step
    implementing `TokenTransformer` (such as an `EntropyTransformer`) before a final
    classifier, or a final `TokenClassifier` (`EPR`, `WEPR`).
    """

    def fit(self, X, y=None, **params) -> "DetectorPipeline":  # ruff: ignore[invalid-argument-name]
        """Fit the pipeline after checking that it can score per token.

        Raises
        ------
        TypeError
            If no step is a `TokenTransformer` before a final classifier and the last step
            is not a `TokenClassifier`.
        """
        if not has_token_layout(self.steps):
            msg = "a DetectorPipeline needs a TokenTransformer step before a classifier, or a final TokenClassifier"
            raise TypeError(msg)
        return super().fit(X, y, **params)

    @beartype(conf=TYPE_CHECKED)
    def predict_token_proba(self, X) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Per-token probabilities of class 1.

        Like `predict_proba`, it raises `NotFittedError` on an unfitted pipeline.

        Parameters
        ----------
        X : array-like
            What the first step accepts, e.g. completion responses for a `LogProbParser`.

        Returns
        -------
        ndarray of shape (n_sequences, max_tokens, 1)
            Probability of class 1 per token, NaN at padded positions.

        Raises
        ------
        TypeError
            If the last step is not a fitted classifier of two classes.
        """
        check_is_fitted(self)
        return token_proba(self.steps, X)


@beartype(conf=TYPE_CHECKED)
def make_detector(*, k: int, classifier: TokenDetector | None = None) -> DetectorPipeline:
    """A pipeline from completion responses to hallucination scores.

    Parameters
    ----------
    k : int
        Candidates per token `LogProbParser` extracts. Required: a published detector is
        fixed at the `k` it was trained with.
    classifier : TokenClassifier, optional
        The detector behind the parser (`EPR`, `WEPR`), untrained to `fit` it or loaded with
        `from_pretrained`. `WEPR()` when None.

    Returns
    -------
    DetectorPipeline
        `LogProbParser(k=k)` then `classifier`, steps named `logprobparser` and the lowercased
        class name. A `Pipeline` with `predict_token_proba`.

    Raises
    ------
    TypeError
        If `classifier` is not a classifier with `predict_token_proba`.

    Examples
    --------
    >>> detector = make_detector(k=15)  # doctest: +SKIP
    """
    steps = make_pipeline(LogProbParser(k=k), WEPR() if classifier is None else classifier).steps
    return DetectorPipeline(steps)
