"""
Pydantic models for representing the data in the generated JSON files.
"""

from collections.abc import Sequence

from pydantic import BaseModel


class TokenLogprob(BaseModel):
    """Represents a single token's log probability."""

    token: str
    logprob: float
    rank: int


class Completion(BaseModel):
    """Represents a single generated completion as a sequence of token logprobs."""

    token_logprobs: dict[int, list[float]]  # Mapping from token position to top-K logprobs


class Result(BaseModel):
    """Represents the full data for a single query."""

    query_id: str
    query: str
    expected_answers: Sequence[str]
    generated_answers: Sequence[dict[str, str]]
    token_logprobs: Sequence[Sequence[Sequence[float]]]


class Dataset(BaseModel):
    """Represents the entire dataset with metadata and results."""

    results: Sequence[Result]
