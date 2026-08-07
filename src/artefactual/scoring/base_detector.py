"""The detector pipeline and the `epr` / `wepr` factories that build it."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from artefactual.exceptions import UncalibratedModelError
from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.utils.io import EstimatorPersistenceMixin, Reduction

# Every published detector was fit at 15 ranks.
DEFAULT_K = 15

# Features each reduction produces: EPR pools the ranks into one, WEPR keeps a mean and a
# max per rank. The loaded estimator is checked against this, since a detector's
# coefficient vector is fixed at the rank count it was trained at.
_FEATURE_COUNT = {"epr": lambda _k: 1, "wepr": lambda k: 2 * k}


def _load_pretrained(reduction: Reduction, identifier: str, k: int) -> BaseEstimator:
    """Load a published detector and check it covers exactly `k` ranks.

    Raises:
        ValueError: If the detector was fit at a different rank count than `k`.
    """
    estimator = BaseDetector.load_estimator(identifier)
    expected = _FEATURE_COUNT[reduction](k)
    # `n_features_in_` is set by fit, so it is absent from the `BaseEstimator` interface
    # even though every estimator reaching here is fitted. Read it through `Any` rather
    # than suppressing per type-checker: the suppression is itself reported as unused by
    # versions that do not raise, which fails the hook the other way round.
    fitted: Any = estimator
    actual: int = fitted.n_features_in_
    if actual != expected:
        implied = actual // 2 if reduction == "wepr" else actual
        msg = (
            f"The {reduction} detector at '{identifier}' takes {actual} feature(s), but "
            f"k={k} needs {expected}. Its coefficients are fixed at the rank count they "
            f"were trained at; pass k={implied}, or use a detector trained at k={k}."
        )
        raise ValueError(msg)
    return estimator


class BaseDetector(Pipeline, EstimatorPersistenceMixin):
    """A `parser -> entropy -> classifier` pipeline returning P(hallucination).

    A scikit-learn `Pipeline`, so `predict`, `predict_proba`, `fit`, `get_params` and
    `clone` behave as expected and the detector composes into `GridSearchCV` and friends.
    Build one with `epr()` or `wepr()` rather than constructing it directly.

    Class 1 is the hallucination class: `predict_proba(...)[:, 1]` is the score of
    interest.
    """

    @property
    def estimator(self) -> BaseEstimator:
        """The final estimator, which is the only fitted step in the pipeline."""
        return self.steps[-1][1]

    def predict_token_proba(self, x) -> np.ndarray:
        """Per-token hallucination probabilities, for locating *where* a response drifts.

        Runs the transformer steps in token mode (`transform_tokens`, falling back to
        `transform`), then scores only the non-padded token rows and scatters the results
        back, so padded positions stay NaN rather than being scored as real tokens.

        Args:
            x: The same input `predict_proba` accepts.

        Returns:
            `(n_sequences, max_tokens, 1)`, NaN at padded positions.
        """
        raw_output = x
        for _, transformer in self.steps[:-1]:
            try:
                step_transform = transformer.transform_tokens
            except AttributeError:
                step_transform = transformer.transform
            raw_output = step_transform(raw_output)

        token_features = raw_output

        n_samples, max_tokens, n_features = token_features.shape
        flat_features = np.asarray(token_features).reshape(n_samples * max_tokens, n_features)
        non_padded = ~np.isnan(flat_features).any(axis=1)

        classifier = self.steps[-1][1]
        flat_scores = np.full(n_samples * max_tokens, np.nan)
        flat_scores[non_padded] = classifier.predict_proba(flat_features[non_padded])[:, 1]

        return flat_scores.reshape(n_samples, max_tokens, 1)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, reduction: str, k: int = DEFAULT_K) -> "BaseDetector":
        """Build a trained detector, selecting the reduction by name.

        Equivalent to calling `epr()` or `wepr()` directly; provided for callers that hold
        the reduction as data.

        Args:
            pretrained_model_name_or_path: A registry model name, or a path to a file.
            reduction: `"epr"` or `"wepr"`.
            k: Rank count the responses carry.

        Returns:
            A detector ready to `predict_proba`.
        """
        factory = {"epr": epr, "wepr": wepr}[reduction]
        return factory(pretrained_model_name_or_path, k=k)

    @classmethod
    def trainable(
        cls, reduction: Reduction, k: int = DEFAULT_K, *, classifier: BaseEstimator | None = None
    ) -> "BaseDetector":
        """An unfitted detector, selecting the reduction by name.

        The counterpart to `from_pretrained` for callers that hold the reduction as data and
        intend to `fit`, so selecting a reduction never means re-deriving the mapping from
        its name to a factory.

        Args:
            reduction: `"epr"` or `"wepr"`.
            k: Rank count the responses carry.
            classifier: Final estimator to fit. Defaults to the unregularised logistic
                regression the published detectors were fit with.

        Returns:
            A detector ready to `fit`.
        """
        factory = {"epr": epr, "wepr": wepr}[reduction]
        return factory(k=k, trainable=True, classifier=classifier)


def _build(
    reduction: Reduction,
    pretrained_model_name_or_path: str | None,
    k: int,
    *,
    trainable: bool,
    classifier: BaseEstimator | None,
    **pipeline_kwargs,
) -> "BaseDetector":
    """Assemble a parser -> entropy -> classifier pipeline pinned to `k` ranks.

    `k` is handled at the ends of the pipeline: the parser sizes the rank axis to it, and
    the classifier checks the loaded weights were trained at it. The entropy step in
    between carries no rank count, since its input width is already `k`.
    """
    if trainable:
        if pretrained_model_name_or_path is not None:
            msg = "Pass either pretrained weights or trainable=True, not both."
            raise ValueError(msg)
        # Unregularised, so the fitted coefficients are comparable to the shipped files.
        # C=np.inf rather than penalty=None: the latter is deprecated in scikit-learn 1.8
        # and removed in 1.10, and the two produce identical coefficients.
        final = classifier if classifier is not None else LogisticRegression(C=np.inf, max_iter=1000)
    else:
        if classifier is not None:
            msg = "`classifier` only applies with trainable=True; pretrained weights bring their own."
            raise ValueError(msg)
        if pretrained_model_name_or_path is None:
            raise UncalibratedModelError()
        final = _load_pretrained(reduction, pretrained_model_name_or_path, k)

    return BaseDetector(
        steps=[
            ("parser", LogProbParser(k=k)),
            ("entropy", EntropyTransformer(reduction=reduction)),
            ("classifier", final),
        ],
        **pipeline_kwargs,
    )


def epr(
    pretrained_model_name_or_path: str | None = None,
    k: int = DEFAULT_K,
    *,
    trainable: bool = False,
    classifier: BaseEstimator | None = None,
    transform_input=None,
    memory=None,
    verbose=False,
) -> "BaseDetector":
    """Build a hallucination detector that pools a response's uncertainty into one number.

    EPR — Entropy Production Rate. A single feature, pooling every rank of the token
    distribution into one number.

    Both estimators need a detector fit on labelled data, so choosing this one saves no
    setup work over `wepr` — only parameters. Prefer `wepr` unless there is too little
    labelled data to fit its larger coefficient vector.

    Example:
        >>> detector = epr("mistralai/Ministral-8B-Instruct-2410")
        >>> detector.predict_proba(response)[:, 1]  # doctest: +SKIP

    Args:
        pretrained_model_name_or_path: A model name from the registry, or a path to a
            detector file. Omit it only together with `trainable=True`.
        k: Rank count the responses carry, and the width EPR averages over. Responses
            carrying fewer than `k` ranks are rejected when parsed.
        trainable: Return an *unfitted* detector to calibrate on your own labelled data.
            Call `fit(responses, y)`, where 1 marks a hallucination.
        classifier: Final estimator, with `trainable=True` only. Defaults to the
            unregularised logistic regression the shipped estimators were fit with.

    Returns:
        A `BaseDetector`, trained and ready to predict, or unfitted if `trainable`.

    Raises:
        UncalibratedModelError: If neither weights nor `trainable=True` were given.
        ValueError: If both were given, or if `classifier` is passed without `trainable`.
    """
    return _build(
        "epr",
        pretrained_model_name_or_path,
        k,
        trainable=trainable,
        classifier=classifier,
        transform_input=transform_input,
        memory=memory,
        verbose=verbose,
    )


def wepr(
    pretrained_model_name_or_path: str | None = None,
    k: int = DEFAULT_K,
    *,
    trainable: bool = False,
    classifier: BaseEstimator | None = None,
    transform_input=None,
    memory=None,
    verbose=False,
) -> "BaseDetector":
    """Build a hallucination detector that reads each rank of the token distribution.

    WEPR — Weighted EPR. `2k` features, one learned coefficient per rank, letting the
    detector weight the informative ranks over the rest.

    The default choice: it costs the same to calibrate as `epr` and reads strictly more of
    the distribution. Fall back to `epr` only when labelled data is too scarce to fit `2k`
    coefficients.

    Example:
        >>> detector = wepr("mistralai/Ministral-8B-Instruct-2410")
        >>> detector.predict_proba(response)[:, 1]  # doctest: +SKIP

    Args:
        pretrained_model_name_or_path: A model name from the registry, or a path to a
            weights file. Omit it only together with `trainable=True`.
        k: Rank count the responses carry. The weights must cover exactly this many ranks,
            and responses carrying fewer are rejected when parsed.
        trainable: Return an *unfitted* detector to calibrate on your own labelled data.
            Call `fit(responses, y)`, where 1 marks a hallucination.
        classifier: Final estimator, with `trainable=True` only. Defaults to the
            unregularised logistic regression the shipped estimators were fit with.

    Returns:
        A `BaseDetector`, trained and ready to predict, or unfitted if `trainable`.

    Raises:
        UncalibratedModelError: If neither weights nor `trainable=True` were given.
        ValueError: If both were given, if `classifier` is passed without `trainable`, or
            if the weights do not cover exactly `k` ranks.
    """
    return _build(
        "wepr",
        pretrained_model_name_or_path,
        k,
        trainable=trainable,
        classifier=classifier,
        transform_input=transform_input,
        memory=memory,
        verbose=verbose,
    )
