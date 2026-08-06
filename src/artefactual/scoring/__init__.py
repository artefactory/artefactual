"""Scoring module for artefactual library.

The public scoring API is a scikit-learn pipeline: `epr()` and `wepr()` build a
`BaseDetector` (a `Pipeline`) exposing the standard `fit`/`transform`/`predict_proba`
surface, plus `predict_token_proba` for token-level scores.
"""

from artefactual.scoring.base_detector import BaseDetector, epr, wepr
from artefactual.scoring.entropy_methods.entropy_contributions import (
    EntropyContributionsMixin,
)
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression

__all__ = [
    "BaseDetector",
    "EntropyContributionsMixin",
    "EntropyTransformer",
    "PretrainedLogisticRegression",
    "epr",
    "wepr",
]
