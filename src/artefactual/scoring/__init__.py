"""Hallucination detectors and the pieces they are built from.

`epr()` and `wepr()` are the entry points: each returns a `BaseDetector`, a scikit-learn
`Pipeline` exposing the standard `fit`/`transform`/`predict_proba` surface plus
`predict_token_proba` for token-level scores. The two differ in how much of a response's
confidence they read; both are used the same way.
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
