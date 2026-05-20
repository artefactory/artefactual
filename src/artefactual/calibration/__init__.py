"""Calibration module for artefactual library."""

import typing

if typing.TYPE_CHECKING:
    from artefactual.calibration.outputs_entropy import GenerationConfig
    from artefactual.calibration.rates_answers import RatingConfig
    from artefactual.calibration.utils.io import load_tqa_from_json, save_to_json
    from artefactual.calibration.utils.memory import clear_gpu_memory
    from artefactual.calibration.utils.models import get_model_name, init_llm

from artefactual.calibration.train_calibration import train_calibration

__all__ = [
    "GenerationConfig",
    "RatingConfig",
    "clear_gpu_memory",
    "get_model_name",
    "init_llm",
    "load_tqa_from_json",
    "save_to_json",
    "train_calibration",
]

def __getattr__(name: str) -> typing.Any:
    if name == "GenerationConfig":
        from artefactual.calibration.outputs_entropy import GenerationConfig
        return GenerationConfig
    if name == "RatingConfig":
        from artefactual.calibration.rates_answers import RatingConfig
        return RatingConfig
    if name == "clear_gpu_memory":
        from artefactual.calibration.utils.memory import clear_gpu_memory
        return clear_gpu_memory
    if name == "get_model_name":
        from artefactual.calibration.utils.models import get_model_name
        return get_model_name
    if name == "init_llm":
        from artefactual.calibration.utils.models import init_llm
        return init_llm
    if name == "load_tqa_from_json":
        from artefactual.calibration.utils.io import load_tqa_from_json
        return load_tqa_from_json
    if name == "save_to_json":
        from artefactual.calibration.utils.io import save_to_json
        return save_to_json
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
