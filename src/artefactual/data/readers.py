"""Utilities for reading and processing data files.

This module provides functions for reading data samples from files and
joining samples from different sources based on a common identifier.
"""

from collections.abc import Callable, Sequence
from typing import NotRequired, TypedDict

import orjson as json
import toolz as tlz
from beartype import beartype
from etils import epath


class DataSample(TypedDict):
    """Common structure for data samples with ID.

    All data samples must have an 'id' field. Other fields are optional
    and can be extended for specific use cases.
    """

    id: str
    question: NotRequired[str]
    answer: NotRequired[str]
    response: NotRequired[str]
    score: NotRequired[float]
    logprobs: NotRequired[list[float]]


# Type for generic dictionary-like objects with string keys
DEFAULT_GET_ID = tlz.curried.get("id")


@beartype
def read_file(path: epath.Path) -> Sequence[DataSample]:
    """Read a file containing JSON lines.

    Reads a file containing one JSON object per line and parses each line
    into a DataSample object.

    Args:
        path: Path to the file to read

    Returns:
        List of data samples parsed from the file

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file contains invalid JSON

    Example:
        >>> samples = read_file(epath.Path("data.jsonl"))
        >>> samples[0]["id"]
        '1'
    """
    with path.open("r") as src:
        lines = src.readlines()
        samples = map(json.loads, lines)
        return list(samples)


@beartype
def join_samples(
    left_samples: Sequence[DataSample],
    right_samples: Sequence[DataSample],
    key_fn: Callable[[DataSample], str] = DEFAULT_GET_ID,
) -> Sequence[DataSample]:
    """Join two sequences of dictionaries on a common key.

    Args:
        left_samples: First sequence of samples
        right_samples: Second sequence of samples
        key_fn: Function to extract the join key from each sample

    Returns:
        List of joined samples

    Raises:
        KeyError: If key_fn fails to extract a key from any sample

    Example:
        >>> ratings = [{"id": "1", "rating": 5}]
        >>> responses = [{"id": "1", "response": "answer"}]
        >>> join_samples(ratings, responses)
        [{"id": "1", "rating": 5, "response": "answer"}]
    """
    if not left_samples or not right_samples:
        return []

    joined = tlz.join(key_fn, right_samples, key_fn, left_samples)
    return [{"id": right["id"], **left, **right} for left, right in joined]
