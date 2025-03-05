"""Utilities for reading data files."""

from collections.abc import Callable, Sequence
from typing import Any, NotRequired, TypedDict, TypeVar

import orjson as json
import toolz as tlz
from beartype import beartype
from etils import epath


class DataSample(TypedDict):
    """Common structure for data samples with ID."""

    id: str
    question: NotRequired[str]
    answer: NotRequired[str]
    response: NotRequired[str]
    score: NotRequired[float]
    logprobs: NotRequired[list[float]]


T = TypeVar("T", bound=dict[str, Any])
DEFAULT_GET_ID = tlz.curried.get("id")


@beartype
def read_file(path: epath.Path) -> list[DataSample]:
    """Read a file containing JSON lines.

    Args:
        path: Path to the file

    Returns:
        List of data samples parsed from the file

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
    ratings: Sequence[DataSample],
    responses: Sequence[DataSample],
    key_fn: Callable[[DataSample], str] = DEFAULT_GET_ID,
) -> list[DataSample]:
    """Join two sequences of dictionaries on a common key.

    Args:
        ratings: First sequence of data samples
        responses: Second sequence of data samples
        key_fn: Function to extract the key from each sample

    Returns:
        List of joined data samples

    Example:
        >>> ratings = [{"id": "1", "rating": 5}]
        >>> responses = [{"id": "1", "response": "answer"}]
        >>> join_samples(ratings, responses)
        [{"id": "1", "rating": 5, "response": "answer"}]
    """
    joined = tlz.join(key_fn, responses, key_fn, ratings)
    return [{"id": left["id"], **left, **right} for left, right in joined]
