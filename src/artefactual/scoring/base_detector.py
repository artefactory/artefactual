import numpy as np
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
    **pipeline_kwargs,
) -> "BaseDetector":
    """Assemble a parser -> entropy -> classifier pipeline pinned to `k` ranks.

    The parser owns the rank axis: it refuses a response carrying fewer than `k` ranks and
    hands on an array exactly `k` wide, so the entropy step needs no rank count of its own
    and the classifier only has to check the weights were calibrated at the same `k`.
    Splitting that ownership is what let a narrow response be zero-filled and scored as
    though its unfetched ranks held no probability mass.
    """
    if pretrained_model_name_or_path is None:
        raise UncalibratedModelError()

    classifier = PretrainedLogisticRegression.from_pretrained(pretrained_model_name_or_path, registry=registry, k=k)
    return BaseDetector(
        steps=[
            ("parser", LogProbParser(k=k)),
            ("entropy", EntropyTransformer(reduction=reduction)),
            ("classifier", classifier),
        ],
        **pipeline_kwargs,
    )


# Factory aliases (module level), each returns a BaseDetector
def epr(
    pretrained_model_name_or_path: str | None = None,
    k: int = DEFAULT_K,
    *,
    transform_input=None,
    memory=None,
    verbose=False,
) -> "BaseDetector":
    """EPR detector. `k` is the top-k rank count the responses will carry."""
    return _build(
        "epr",
        "calibration",
        pretrained_model_name_or_path,
        k,
        transform_input=transform_input,
        memory=memory,
        verbose=verbose,
    )


def wepr(
    pretrained_model_name_or_path: str | None = None,
    k: int = DEFAULT_K,
    *,
    transform_input=None,
    memory=None,
    verbose=False,
) -> "BaseDetector":
    """WEPR detector. `k` is the top-k rank count; the weights must be calibrated at it."""
    return _build(
        "wepr",
        "weights",
        pretrained_model_name_or_path,
        k,
        transform_input=transform_input,
        memory=memory,
        verbose=verbose,
    )
