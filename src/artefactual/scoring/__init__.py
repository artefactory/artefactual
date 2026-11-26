"""Hallucination detectors and the pieces they are built from.

`EPR` and `WEPR` score logprob arrays and are plain scikit-learn classifiers: `fit`,
`predict_proba`, `clone` and `GridSearchCV` behave as expected, and `predict_token_proba`
gives token-level scores. To score completion responses, put a
`LogProbParser` in front:

    detector = make_detector(k=15, classifier=WEPR.from_pretrained("artefactory/wepr-phi4"))
    detector = make_detector(k=15).fit(responses, y)
"""

from artefactual.scoring.entropy_classifier import EPR, WEPR
from artefactual.scoring.entropy_methods.entropy_contributions import (
    EntropyContributionsMixin,
)
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.entropy_methods.pooling import (
    EPRPooling,
    FunctionPooling,
    Pooling,
    WEPRPooling,
    epr_reduction,
    wepr_reduction,
)
from artefactual.scoring.make_detector import DetectorPipeline, make_detector
from artefactual.scoring.tokens import TokenClassifier, TokenProbaMixin, TokenTransformer, token_proba

__all__ = [
    "EPR",
    "WEPR",
    "DetectorPipeline",
    "EPRPooling",
    "EntropyContributionsMixin",
    "EntropyTransformer",
    "FunctionPooling",
    "Pooling",
    "TokenClassifier",
    "TokenProbaMixin",
    "TokenTransformer",
    "WEPRPooling",
    "epr_reduction",
    "make_detector",
    "token_proba",
    "wepr_reduction",
]
