# src/artefactual/utils/weights.py
import importlib.resources
import json
from pathlib import Path

# The registry map
MODEL_WEIGHT_MAP = {
    "tiiuae/Falcon3-10B-Instruct": "weights_Falcon3.json",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "weights_Small.json",
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
            with open(local_path, encoding="utf-8") as f:
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
