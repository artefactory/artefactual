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
    def predict_token_proba(self, x) -> np.ndarray:
        """Token-level hallucination probabilities, shape (n_samples, max_tokens, 1).

        Routes the pipeline in token mode (transform_tokens, with transform fallback),
        mask padded (NaN) token rows before the classifier, scatter results back so
        padded positions stay NaN.
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
        """Build a ready-to-predict detector with weights loaded."""
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

    The parser owns the rank axis: it refuses a response carrying fewer than `k` ranks and
    hands on an array exactly `k` wide, so the entropy step needs no rank count of its own
    and the classifier only has to check the weights were calibrated at the same `k`.
    Splitting that ownership is what let a narrow response be zero-filled and scored as
    though its unfetched ranks held no probability mass.
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


# Factory aliases (module level), each returns a BaseDetector
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
    """EPR detector.

    Args:
        pretrained_model_name_or_path: A model name from the registry, or a path to a
            calibration file. Omit it only together with `trainable=True`.
        k: The top-k rank count the responses carry. EPR is a mean over ranks, so
            this is part of the feature definition.
        trainable: Return an *unfitted* detector to calibrate on your own labelled data.
            Call `fit(responses, y)`, where 1 marks a hallucination.
        classifier: Final estimator, with `trainable=True` only. Defaults to the
            unregularised logistic regression the shipped calibrations were fit with.

    Raises:
        UncalibratedModelError: If neither weights nor `trainable=True` were given.
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
    """WEPR detector.

    Args:
        pretrained_model_name_or_path: A model name from the registry, or a path to a
            calibration file. Omit it only together with `trainable=True`.
        k: The top-k rank count the responses carry. The weights must be calibrated at it.
        trainable: Return an *unfitted* detector to calibrate on your own labelled data.
            Call `fit(responses, y)`, where 1 marks a hallucination.
        classifier: Final estimator, with `trainable=True` only. Defaults to the
            unregularised logistic regression the shipped calibrations were fit with.

    Raises:
        UncalibratedModelError: If neither weights nor `trainable=True` were given.
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
