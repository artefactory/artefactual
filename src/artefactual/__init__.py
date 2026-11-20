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
    BaseDetector,
    BaseEntropyClassifier,
    EntropyTransformer,
    EPRClassifier,
    EPRPooling,
    FunctionPooling,
    Pooling,
    PoolingDetector,
    WEPRClassifier,
    WEPRPooling,
    epr_reduction,
    wepr_reduction,
)

__all__ = [
    "EPR",
    "WEPR",
    "BaseDetector",
    "BaseEntropyClassifier",
    "BatchRequestOutput",
    "ChatCompletion",
    "EPRClassifier",
    "EPRPooling",
    "EntropyTransformer",
    "FunctionPooling",
    "Judgment",
    "LogProbParser",
    "Pooling",
    "PoolingDetector",
    "ResponsesPayload",
    "WEPRClassifier",
    "WEPRPooling",
    "__version__",
    "epr_reduction",
    "read_batch",
    "wepr_reduction",
]
