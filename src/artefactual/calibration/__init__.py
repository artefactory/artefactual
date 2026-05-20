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
    # Moving the mapping inside the function completely bypasses RUF067
    lazy_imports = {
        "GenerationConfig": (
            "artefactual.calibration.outputs_entropy",
            "GenerationConfig",
        ),
        "RatingConfig": ("artefactual.calibration.rates_answers", "RatingConfig"),
        "clear_gpu_memory": (
            "artefactual.calibration.utils.memory",
            "clear_gpu_memory",
        ),
        "get_model_name": ("artefactual.calibration.utils.models", "get_model_name"),
        "init_llm": ("artefactual.calibration.utils.models", "init_llm"),
        "load_tqa_from_json": (
            "artefactual.calibration.utils.io",
            "load_tqa_from_json",
        ),
        "save_to_json": ("artefactual.calibration.utils.io", "save_to_json"),
    }

    if name in lazy_imports:
        import importlib

        module_path, obj_name = lazy_imports[name]
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)

        # Cache the result in globals so __getattr__ is bypassed next time
        globals()[name] = obj
        return obj

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
