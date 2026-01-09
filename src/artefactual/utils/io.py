import importlib.resources
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# calibration and weights loading utilities
MODEL_WEIGHT_MAP = {
    "tiiuae/Falcon3-10B-Instruct": "weights_falcon3.json",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "weights_mistral_small.json",
    "mistralai/Ministral-8B-Instruct-2410": "weights_ministral.json",
    "microsoft/phi-4": "weights_phi4.json",
}


def load_weights(identifier: str) -> dict[str, float]:
    """
    Loads weights from a built-in model name or a local file path.

    Args:
        identifier: Either a built-in model name
        (as listed in MODEL_WEIGHT_MAP) or a path to a local JSON file containing weights.

    Returns:
        A dictionary mapping string keys to float values representing the weights.

    Raises:
        ValueError: If the identifier is not a supported model name,
        the file does not exist, or the file is not valid JSON.
    """
    # 1. Check if it matches a built-in model name (The Registry)
    if identifier in MODEL_WEIGHT_MAP:
        filename = MODEL_WEIGHT_MAP[identifier]
        package_files = importlib.resources.files("artefactual.data")
        with package_files.joinpath(filename).open("r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Check if it is a valid path to a user file on disk
    local_path = Path(identifier)
    if local_path.is_file():
        try:
            with Path(local_path).open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as err:
            msg = f"The file at '{identifier}' is not valid JSON."
            raise ValueError(msg) from err

    # 3. If neither, raise an error listing available options
    available = ", ".join(MODEL_WEIGHT_MAP.keys())
    msg = (
        f"Could not find weights for '{identifier}'. "
        f"Ensure it is a valid file path OR one of the supported models: {available}"
    )
    raise ValueError(msg)


# The registry map for calibration files
MODEL_CALIBRATION_MAP = {
    "tiiuae/Falcon3-10B-Instruct": "calibration_falcon3.json",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "calibration_mistral_small.json",
    "mistralai/Ministral-8B-Instruct-2410": "calibration_ministral.json",
    "microsoft/phi-4": "calibration_phi4.json",
    # Add other models here as needed
}


def load_calibration(identifier: str) -> dict[str, float]:
    """
    Loads calibration coefficients from a built-in model name OR a local file path.

    Args:
        identifier: The model name (e.g., "tiiuae/Falcon3-10B-Instruct") or a path to a JSON file.

    Returns:
        A dictionary containing the calibration coefficients (e.g., "intercept", "coefficients").

    Raises:
        ValueError: If the identifier is not found in the registry and is not a valid file path,
                    or if the file is not valid JSON.
    """
    # 1. Check if it matches a built-in model name (The Registry)
    if identifier in MODEL_CALIBRATION_MAP:
        filename = MODEL_CALIBRATION_MAP[identifier]
        package_files = importlib.resources.files("artefactual.data")
        with package_files.joinpath(filename).open("r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Check if it is a valid path to a user file on disk
    local_path = Path(identifier)
    if local_path.is_file():
        try:
            with Path(local_path).open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as err:
            msg = f"The file at '{identifier}' is not valid JSON."
            raise ValueError(msg) from err

    # 3. If neither, raise an error listing available options
    available = ", ".join(MODEL_CALIBRATION_MAP.keys())
    msg = (
        f"Could not find calibration for '{identifier}'. "
        f"Ensure it is a valid file path OR one of the supported models: {available}"
    )
    raise ValueError(msg)
