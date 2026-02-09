"""Scoring module for artefactual library.

This module re-exports commonly used scoring classes and helpers so they can be
imported from `artefactual.scoring` instead of deep submodules.
"""

from artefactual.scoring.entropy_methods.entropy_contributions import compute_entropy_contributions
from artefactual.scoring.entropy_methods.epr import EPR
from artefactual.scoring.entropy_methods.wepr import WEPR
from artefactual.scoring.probability_methods.sentence_proba import SentenceProbabilityScorer
from artefactual.scoring.uncertainty_detector import (
    LogProbUncertaintyDetector,
    UncertaintyDetector,
)

__all__ = [
    "EPR",
    "WEPR",
    "LogProbUncertaintyDetector",
    "SentenceProbabilityScorer",
    "UncertaintyDetector",
    "compute_entropy_contributions",
]
