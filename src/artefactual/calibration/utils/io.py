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
    """
    Load the pack data from a JSON file.

    Args:
        input_file (str): Path to the JSON file

    Returns:
        List of (question, question_id, short_answer, answer_aliases) tuples.
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
