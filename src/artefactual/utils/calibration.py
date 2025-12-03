import importlib.resources
import json
from pathlib import Path

# The registry map for calibration files
MODEL_CALIBRATION_MAP = {
    "tiiuae/Falcon3-10B-Instruct": "calibration_Falcon3.json",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "calibration_Small.json",
    "mistralai/Ministral-8B-Instruct-2410": "calibration_Ministral.json",
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
            with open(local_path, encoding="utf-8") as f:
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
