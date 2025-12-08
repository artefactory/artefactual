import importlib.resources
import json
import logging
from pathlib import Path
from typing import Any


def save_to_json(data: list[dict[str, Any]], output_file: str):
    """Save data to a JSON file.

    Args:
        data (list[dict[str, Any]]): Data to save.
        output_file (str): Path to the output JSON file.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved {len(data)} items to {output_file}")
    except OSError as e:
        logging.error(f"Error writing to {output_file}: {e}")


def load_tqa_from_json(
    input_file: str,
) -> list[tuple[str, str, str, list[str]]]:
    """Load the pack data from a JSON file.

    Args:
        input_file (str): Path to the JSON file

    Returns:
        list: List of (question, answer) tuples
    """
    try:
        with open(input_file, encoding="utf-8") as f:
            json_data = json.load(f)

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
        logging.info(f"Loaded {len(pack_data)} question-answer pairs from {input_file}")
        return pack_data
    except FileNotFoundError:
        logging.error(f"File not found: {input_file}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {input_file}")
        return []
    except KeyError as e:
        logging.error(f"Missing key in JSON data: {e}")
        return []


def convert_bytes_to_str(obj):
    """Recursively convert bytes to strings in a nested structure.

    Args:
        obj: The object to process. Can be bytes, dict, list, or other types.

    Returns:
        The object with bytes converted to UTF-8 strings.
    """
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception as e:
            logging.error(f"Error decoding bytes: {e}")
            return "ERROR"
    elif isinstance(obj, dict):
        return {key: convert_bytes_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_bytes_to_str(element) for element in obj]
    return obj


# calibration and weights loading utilities
# For now the model map stays here, but could be moved to a config file later ?
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
