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
    DetectorPipeline,
    EntropyTransformer,
    EPRPooling,
    FunctionPooling,
    Pooling,
    TokenClassifier,
    TokenProbaMixin,
    TokenTransformer,
    WEPRPooling,
    epr_reduction,
    make_detector,
    token_proba,
    wepr_reduction,
)

__all__ = [
    "EPR",
    "WEPR",
    "BatchRequestOutput",
    "ChatCompletion",
    "DetectorPipeline",
    "EPRPooling",
    "EntropyTransformer",
    "FunctionPooling",
    "Judgment",
    "LogProbParser",
    "Pooling",
    "ResponsesPayload",
    "TokenClassifier",
    "TokenProbaMixin",
    "TokenTransformer",
    "WEPRPooling",
    "__version__",
    "epr_reduction",
    "make_detector",
    "read_batch",
    "token_proba",
    "wepr_reduction",
]
