"""Calibration module for artefactual library."""

from artefactual.calibration.train_calibration import train_calibration

__all__ = ["train_calibration"]

try:
    from artefactual.calibration.outputs_entropy import GenerationConfig
    from artefactual.calibration.rates_answers import RatingConfig
    from artefactual.calibration.utils.io import load_tqa_from_json, save_to_json
    from artefactual.calibration.utils.memory import clear_gpu_memory
    from artefactual.calibration.utils.models import get_model_name, init_llm
except ImportError:
    pass
else:
    __all__ += [
        "GenerationConfig",
        "RatingConfig",
        "clear_gpu_memory",
        "get_model_name",
        "init_llm",
        "load_tqa_from_json",
        "save_to_json",
    ]
