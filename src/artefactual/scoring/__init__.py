"""Hallucination detectors and the pieces they are built from.

`EPR` and `WEPR` are the entry points. Each is a scikit-learn `Pipeline` exposing the
standard `fit`/`transform`/`predict_proba` surface plus `predict_token_proba` for
token-level scores, and the two differ only in how much of a response's confidence they
read. Construct one to fit your own weights; call `from_pretrained` on it to score with
published ones.

    detector = WEPR.from_pretrained("artefactory/wepr-phi4")
    detector = WEPR(k=15).fit(responses, y)
"""

from artefactual.scoring.base_detector import EPR, WEPR, BaseDetector
from artefactual.scoring.entropy_methods.entropy_contributions import (
    EntropyContributionsMixin,
)
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer

__all__ = [
    "EPR",
    "WEPR",
    "BaseDetector",
    "EntropyContributionsMixin",
    "EntropyTransformer",
]
