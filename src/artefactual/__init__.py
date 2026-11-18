"""Artefactual: hallucination detection for black-box LLMs, as scikit-learn estimators."""

from artefactual._version import __version__
from artefactual.scoring import EPR, WEPR, EPRClassifier, WEPRClassifier

__all__ = ["EPR", "WEPR", "EPRClassifier", "WEPRClassifier", "__version__"]
