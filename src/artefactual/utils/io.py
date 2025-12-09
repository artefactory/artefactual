import importlib.resources
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_to_json(data: dict[str, Any] | list[dict[str, Any]], output_file: str) -> None:
    """Save data to a JSON file.

    Supports two common use cases:
    - Dataset-level outputs: A single dict with metadata and results (e.g., entropy datasets)
    - Item-level outputs: A list of individual item dicts

    Args:
        data: Data to save (dict or list of dicts).
        output_file: Path to the output JSON file.
    """
    output_path = Path(output_file)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        if isinstance(data, list):
            logger.info("Saved %d items to %s", len(data), output_path)
        else:
            logger.info("Saved dataset to %s", output_path)
    except OSError:
        logger.exception("Error writing to %s", output_path)


def load_tqa_from_json(
    input_file: str,
) -> list[tuple[str, str, str, list[str]]]:
    """Load the pack data from a JSON file.

    Args:
        input_file (str): Path to the JSON file

    Returns:
        list: List of (question, answer) tuples
    """
    input_path = Path(input_file)
    try:
        with input_path.open(encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        logger.exception("File not found: %s", input_path)
        return []
    except json.JSONDecodeError:
        logger.exception("Error decoding JSON from %s", input_path)
        return []

    try:
        # Convert dictionaries back to tuples
        pack_data = [
            (
                item["question"],
                item["question_id"],
                item["short_answer"],
                item["answer_aliases"],
            )
            for item in json_data
        ]
    except KeyError:
        logger.exception("Missing key in JSON data")
        return []

    logger.info("Loaded %s question-answer pairs from %s", len(pack_data), input_path)
    return pack_data


def convert_bytes_to_str(obj: Any) -> Any:
    """Recursively convert bytes to strings in a nested structure.

    Args:
        obj: The object to process. Can be bytes, dict, list, or other types.

    Returns:
        The object with bytes converted to UTF-8 strings.
    """
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            logger.exception("Error decoding bytes")
            return "ERROR"
    elif isinstance(obj, dict):
        return {key: convert_bytes_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_bytes_to_str(element) for element in obj]
    return obj


# calibration and weights loading utilities
# For now the model map stays here, but could be moved to a config file later ?
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
