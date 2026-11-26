"""Per-token scores: the protocols a pipeline step implements and the walk that chains them."""

from typing import Annotated, Any, Protocol, runtime_checkable

import numpy as np
from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is
from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted

from artefactual.utils.typing import TYPE_CHECKED, LogprobArray


@runtime_checkable
class TokenTransformer(Protocol):
    """A step that also maps its input to one feature row per token."""

    def transform(self, X: Any) -> Any:  # ruff: ignore[invalid-argument-name]
        """Features per sequence."""

    def transform_tokens(self, X: Any) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Features per token, shape `(n_sequences, n_tokens, n_features)`, NaN at padding."""


@runtime_checkable
class TokenClassifier(Protocol):
    """A classifier that scores each token of a sequence itself."""

    def predict_proba(self, X: Any) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Class probabilities per sequence."""

    def predict_token_proba(self, X: Any) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Probabilities of class 1 per token, shape `(n_sequences, n_tokens, 1)`."""

    def __sklearn_tags__(self) -> Any:
        """The scikit-learn tags."""


def is_fitted_binary(classifier: Any) -> bool:
    """Whether `classifier.classes_` is an array of two labels.

    A wrapper that forwards attributes, such as `FrozenEstimator`, counts as the classifier it wraps.
    """
    try:
        classes = classifier.classes_
    except AttributeError:
        return False
    return is_bearable(classes, np.ndarray) and len(classes) == 2


FittedBinaryClassifier = Annotated[Any, Is[is_fitted_binary]]


@beartype(conf=TYPE_CHECKED)
def require_fitted_binary(classifier: FittedBinaryClassifier) -> None:
    """Refuse a last step that is not a fitted classifier of two classes."""


def has_token_layout(steps: list) -> bool:
    """Whether `token_proba(steps, X)` can score: a token step before a classifier, or a last `TokenClassifier`."""
    *head, (_, last) = steps
    has_token_step = any(is_bearable(step, TokenTransformer) for _, step in head)
    return (has_token_step and is_classifier(last)) or is_bearable(last, TokenClassifier)


def token_proba(steps: list, X: Any) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
    """Per-token probabilities of class 1 through fitted `(name, estimator)` steps.

    Each step before the last runs `transform_tokens` if it implements `TokenTransformer`,
    else `transform`. The first token step defines the tokens: its rows are flattened to one
    per token and the padded ones are dropped, so later steps and the last step's
    `predict_proba` see real tokens only. Without a token step, the last step's own
    `predict_token_proba` answers when it is a `TokenClassifier`.

    Parameters
    ----------
    steps : list of (str, estimator)
        Fitted steps, as `Pipeline.steps`.
    X : array-like
        What the first step accepts.

    Returns
    -------
    ndarray of shape (n_sequences, n_tokens, 1)
        Probability of class 1 per token, NaN at padded positions; `(0, 0, 1)` for an empty batch.

    Raises
    ------
    TypeError
        If the last step is not a fitted classifier of two classes, or if no step is a
        `TokenTransformer` and the last is not a `TokenClassifier`.
    """
    *head, (_, last) = steps
    require_fitted_binary(last)
    shape: tuple[int, int] = (0, 0)
    keep: np.ndarray | None = None
    current = X
    for _, step in head:
        if keep is None and is_bearable(step, TokenTransformer):
            tokens = step.transform_tokens(current)
            n_sequences, n_tokens, n_features = tokens.shape
            shape = (n_sequences, n_tokens)
            tokens = tokens.reshape(n_sequences * n_tokens, n_features)
            keep = ~np.isnan(tokens).any(axis=1)
            current = tokens[keep]
        else:
            current = step.transform(current)
    if keep is None:
        if is_bearable(last, TokenClassifier):
            return last.predict_token_proba(current)
        msg = "no step implements transform_tokens and the last step has no predict_token_proba"
        raise TypeError(msg)
    scores = np.full(shape[0] * shape[1], np.nan)
    if keep.any():  # classifiers refuse zero rows, which an empty or fully padded batch gives
        scores[keep] = last.predict_proba(current)[:, 1]
    return scores.reshape(*shape, 1)


class TokenProbaMixin:
    """`predict_token_proba` for a classifier that builds a fitted `pipeline_`.

    The class sets `pipeline_`, a fitted `Pipeline`, in `fit`. `predict_token_proba` scores
    through `token_proba(self.pipeline_.steps, X)`.

    `X` is a logprob array, as `predict_proba` takes it. A completion response raises
    `TypeError` naming `is_logprob_array`. Responses are scored through `make_detector` or
    `DetectorPipeline`, which parse them into logprob arrays first.
    """

    @beartype(conf=TYPE_CHECKED)
    def predict_token_proba(self, X: LogprobArray) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Per-token probabilities of class 1, for locating *where* a response drifts.

        Parameters
        ----------
        X : array-like
            The same input `predict_proba` accepts: a logprob array of shape
            (n_sequences, n_tokens, width) or (n_sequences, width).

        Returns
        -------
        ndarray of shape (n_sequences, max_tokens, 1)
            Probability of class 1 per token, NaN at padded positions; `(0, 0, 1)` for an empty batch.

        Raises
        ------
        NotFittedError
            If the estimator is not fitted.
        TypeError
            If `X` holds completion responses, if the class has no `pipeline_`, or from
            `token_proba` if the pipeline cannot score tokens or its last step is not a
            fitted classifier of two classes.
        """
        check_is_fitted(self)
        try:
            steps = self.pipeline_.steps  # ty: ignore[missing-attribute]
        except AttributeError as error:
            msg = f"{type(self).__name__} has no fitted `pipeline_`, which `predict_token_proba` scores through"
            raise TypeError(msg) from error
        return token_proba(steps, X)
