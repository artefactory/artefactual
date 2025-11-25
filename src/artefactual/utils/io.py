import json
import logging
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
