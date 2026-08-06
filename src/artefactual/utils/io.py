"""Resolution of calibration and weight files, by model name or by path.

Two registries, keyed by the same model names: `MODEL_WEIGHT_MAP` for WEPR weights and
`MODEL_CALIBRATION_MAP` for EPR calibrations. A name is resolved against the files shipped
in `artefactual.data`; anything that is a readable path is loaded directly, so callers can
mix published and locally fitted calibrations.
"""

import importlib.resources
import json
from pathlib import Path
from typing import Any

MODEL_WEIGHT_MAP = {
    "tiiuae/Falcon3-10B-Instruct": "weights_falcon3.json",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "weights_mistral_small.json",
    "mistralai/Ministral-8B-Instruct-2410": "weights_ministral.json",
    "microsoft/phi-4": "weights_phi4.json",
}

MODEL_CALIBRATION_MAP = {
    "tiiuae/Falcon3-10B-Instruct": "calibration_falcon3.json",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "calibration_mistral_small.json",
    "mistralai/Ministral-8B-Instruct-2410": "calibration_ministral.json",
    "microsoft/phi-4": "calibration_phi_4.json",
}


def _read_json(source: Any, identifier: str) -> dict[str, Any]:
    """Parse JSON from an open-able source, naming the file if it does not parse.

    Shipped resources and user paths both come through here, so a corrupt file reports the
    same way whichever it was.
    """
    try:
        with source.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as err:
        msg = f"The file at '{identifier}' is not valid JSON."
        raise ValueError(msg) from err


def _load(identifier: str | Path, registry: dict[str, str], kind: str) -> dict[str, Any]:
    """Resolve *identifier* against *registry*, then the filesystem.

    Args:
        identifier: A model name in *registry*, or a path to a JSON file.
        registry: Model name -> filename shipped in `artefactual.data`.
        kind: Word naming what is being loaded, used in the error message.

    Returns:
        The parsed file: an "intercept" float and a "coefficients" mapping.

    Raises:
        ValueError: If *identifier* is neither a registry key nor a readable file, or if
            the file it names does not parse.
    """
    identifier = str(identifier)

    if identifier in registry:
        shipped = importlib.resources.files("artefactual.data").joinpath(registry[identifier])
        return _read_json(shipped, identifier)

    local_path = Path(identifier)
    if local_path.is_file():
        return _read_json(local_path, identifier)

    available = ", ".join(registry)
    msg = (
        f"Could not find {kind} for '{identifier}'. "
        f"Ensure it is a valid file path OR one of the supported models: {available}"
    )
    raise ValueError(msg)


def load_weights(identifier: str | Path) -> dict[str, Any]:
    """Load WEPR weights from a built-in model name or a local file path.

    Args:
        identifier: A model name listed in `MODEL_WEIGHT_MAP`, or a path to a JSON file.

    Returns:
        A dictionary holding an "intercept" float and a "coefficients" mapping.

    Raises:
        ValueError: If the identifier is neither a supported model name nor a readable
            file, or if the file is not valid JSON.
    """
    return _load(identifier, MODEL_WEIGHT_MAP, "weights")


def load_calibration(identifier: str | Path) -> dict[str, Any]:
    """Load an EPR calibration from a built-in model name or a local file path.

    Args:
        identifier: A model name listed in `MODEL_CALIBRATION_MAP`, or a path to a JSON
            file.

    Returns:
        A dictionary with an "intercept" float and a "coefficients" mapping -- the same
        envelope `load_weights` returns, so both registries are interchangeable to callers.

    Raises:
        ValueError: If the identifier is neither a supported model name nor a readable
            file, or if the file is not valid JSON.
    """
    return _load(identifier, MODEL_CALIBRATION_MAP, "calibration")
