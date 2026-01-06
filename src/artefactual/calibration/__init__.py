"""Calibration module for artefactual library.

This module exports configuration classes and training functions for calibration.
"""

from artefactual.calibration.helpers.memory import clear_gpu_memory
from artefactual.calibration.helpers.models import get_model_name, init_llm
from artefactual.calibration.outputs_entropy import GenerationConfig
from artefactual.calibration.rates_answers import RatingConfig
from artefactual.calibration.train_calibration import train_calibration

__all__ = [
    "GenerationConfig",
    "RatingConfig",
    "clear_gpu_memory",
    "get_model_name",
    "init_llm",
    "train_calibration",
]
