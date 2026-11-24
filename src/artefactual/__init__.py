"""Artefactual: hallucination detection for black-box LLMs, as scikit-learn estimators."""

from artefactual._version import __version__
from artefactual.preprocessing import (
    BatchRequestOutput,
    ChatCompletion,
    Judgment,
    LogProbParser,
    ResponsesPayload,
    read_batch,
)
from artefactual.scoring import (
    EPR,
    WEPR,
    EntropyTransformer,
    EPRClassifier,
    WEPRClassifier,
    epr_reduction,
    wepr_reduction,
)

__all__ = [
    "EPR",
    "WEPR",
    "BatchRequestOutput",
    "ChatCompletion",
    "EPRClassifier",
    "EntropyTransformer",
    "Judgment",
    "LogProbParser",
    "ResponsesPayload",
    "WEPRClassifier",
    "__version__",
    "epr_reduction",
    "read_batch",
    "wepr_reduction",
]
