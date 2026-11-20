"""Artefactual: hallucination detection for black-box LLMs, as scikit-learn estimators."""

from artefactual._version import __version__
from artefactual.preprocessing import Judgment
from artefactual.scoring import EPR, WEPR, EPRClassifier, WEPRClassifier

__all__ = ["EPR", "WEPR", "EPRClassifier", "Judgment", "WEPRClassifier", "__version__"]
