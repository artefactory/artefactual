"""Artefactual: hallucination detection for black-box LLMs, as scikit-learn estimators."""

from artefactual._version import __version__
from artefactual.preprocessing import BatchRequestOutput, ChatCompletion, Judgment, ResponsesPayload
from artefactual.scoring import EPR, WEPR, EPRClassifier, WEPRClassifier

__all__ = [
    "EPR",
    "WEPR",
    "BatchRequestOutput",
    "ChatCompletion",
    "EPRClassifier",
    "Judgment",
    "ResponsesPayload",
    "WEPRClassifier",
    "__version__",
]
