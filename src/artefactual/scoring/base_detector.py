import numpy as np
from sklearn.pipeline import Pipeline

from artefactual.exceptions import UncalibratedModelError
from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression


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
    def from_pretrained(cls, pretrained_model_name_or_path: str, reduction: str) -> "BaseDetector":
        """Build a ready-to-predict detector with weights loaded."""
        classifier = PretrainedLogisticRegression.from_pretrained(pretrained_model_name_or_path)
        return cls(
            steps=[
                ("parser", LogProbParser()),
                ("entropy", EntropyTransformer(reduction=reduction)),
                ("classifier", classifier),
            ]
        )


# Factory aliases (module level), each returns a BaseDetector
def epr(
    pretrained_model_name_or_path: str | None = None, *, transform_input=None, memory=None, verbose=False
) -> "BaseDetector":

    if pretrained_model_name_or_path is not None:
        classifier = PretrainedLogisticRegression.from_pretrained(pretrained_model_name_or_path)
    else:
        raise UncalibratedModelError()

    return BaseDetector(
        steps=[
            ("parser", LogProbParser()),
            ("entropy", EntropyTransformer(reduction="epr")),
            ("classifier", classifier),
        ],
        transform_input=transform_input,
        memory=memory,
        verbose=verbose,
    )


def wepr(
    pretrained_model_name_or_path: str | None = None, *, transform_input=None, memory=None, verbose=False
) -> "BaseDetector":

    if pretrained_model_name_or_path is not None:
        classifier = PretrainedLogisticRegression.from_pretrained(pretrained_model_name_or_path)
    else:
        raise UncalibratedModelError()

    return BaseDetector(
        steps=[
            ("parser", LogProbParser()),
            ("entropy", EntropyTransformer(reduction="wepr")),
            ("classifier", classifier),
        ],
        transform_input=transform_input,
        memory=memory,
        verbose=verbose,
    )
