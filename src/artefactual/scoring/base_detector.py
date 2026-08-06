"""The detector pipeline and the `epr` / `wepr` factories that build it."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from artefactual.exceptions import UncalibratedModelError
from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression, Registry

# Every calibration and weights file shipped with the package was fit at 15 ranks.
DEFAULT_K = 15


class BaseDetector(Pipeline):
    """A `parser -> entropy -> classifier` pipeline returning P(hallucination).

    A scikit-learn `Pipeline`, so `predict`, `predict_proba`, `fit`, `get_params` and
    `clone` behave as expected and the detector composes into `GridSearchCV` and friends.
    Build one with `epr()` or `wepr()` rather than constructing it directly.

    Class 1 is the hallucination class: `predict_proba(...)[:, 1]` is the score of
    interest.
    """

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
        """Build a calibrated detector, selecting the reduction by name.

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


def _build(
    reduction: str,
    registry: Registry,
    pretrained_model_name_or_path: str | None,
    k: int,
    *,
    trainable: bool,
    classifier: BaseEstimator | None,
    **pipeline_kwargs,
) -> "BaseDetector":
    """Assemble a parser -> entropy -> classifier pipeline pinned to `k` ranks.

    `k` is handled at the ends of the pipeline: the parser sizes the rank axis to it, and
    the classifier checks the loaded weights were calibrated at it. The entropy step in
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
        final = PretrainedLogisticRegression.from_pretrained(pretrained_model_name_or_path, registry=registry, k=k)

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
    """Build an EPR detector: Entropy Production Rate, pooled over ranks and tokens.

    One feature — the mean entropy contribution per rank, averaged over tokens. Cheaper
    and lower-variance than WEPR, but it cannot weight ranks differently.

    Example:
        >>> detector = epr("mistralai/Ministral-8B-Instruct-2410")
        >>> detector.predict_proba(response)[:, 1]  # doctest: +SKIP

    Args:
        pretrained_model_name_or_path: A model name from the registry, or a path to a
            calibration file. Omit it only together with `trainable=True`.
        k: Rank count the responses carry, and the width EPR averages over. Responses
            carrying fewer than `k` ranks are rejected when parsed.
        trainable: Return an *unfitted* detector to calibrate on your own labelled data.
            Call `fit(responses, y)`, where 1 marks a hallucination.
        classifier: Final estimator, with `trainable=True` only. Defaults to the
            unregularised logistic regression the shipped calibrations were fit with.

    Returns:
        A `BaseDetector`, calibrated and ready to predict, or unfitted if `trainable`.

    Raises:
        UncalibratedModelError: If neither weights nor `trainable=True` were given.
        ValueError: If both were given, or if `classifier` is passed without `trainable`.
    """
    return _build(
        "epr",
        "calibration",
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
    """Build a WEPR detector: Weighted EPR, one learned coefficient per rank.

    `2k` features — the per-rank mean and peak across tokens. Because `-p*log(p)` peaks at
    `p = 1/e`, mid-ranked candidates carry more signal than the near-certain top rank or
    the negligible tail, and weighting ranks separately lets the calibration exploit that.

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
            unregularised logistic regression the shipped calibrations were fit with.

    Returns:
        A `BaseDetector`, calibrated and ready to predict, or unfitted if `trainable`.

    Raises:
        UncalibratedModelError: If neither weights nor `trainable=True` were given.
        ValueError: If both were given, if `classifier` is passed without `trainable`, or
            if the weights do not cover exactly `k` ranks.
    """
    return _build(
        "wepr",
        "weights",
        pretrained_model_name_or_path,
        k,
        trainable=trainable,
        classifier=classifier,
        transform_input=transform_input,
        memory=memory,
        verbose=verbose,
    )
