"""Hallucination detectors and the pieces they are built from.

`epr()` and `wepr()` are the entry points: each returns an unfitted `BaseDetector`, a
scikit-learn `Pipeline` exposing the standard `fit`/`transform`/`predict_proba` surface
plus `predict_token_proba` for token-level scores. The two differ in how much of a
response's confidence they read; both are used the same way.

`BaseDetector.from_pretrained` is the other way in: it returns a detector already carrying
published weights, for scoring without fitting anything.
"""

from artefactual.scoring.base_detector import BaseDetector, epr, wepr
from artefactual.scoring.entropy_methods.entropy_contributions import (
    EntropyContributionsMixin,
)
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer

__all__ = [
    "BaseDetector",
    "EntropyContributionsMixin",
    "EntropyTransformer",
    "epr",
    "wepr",
]
