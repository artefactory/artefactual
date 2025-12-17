"""
Pydantic models for representing the data in the generated JSON files.
"""

from collections.abc import Sequence

from pydantic import BaseModel


class TokenLogprob(BaseModel):
    """
    Represents a single token's log probability.

    Attributes:
        token: The token string.
        logprob: The log probability of the token.
        rank: The rank of the token in the probability distribution.
    """

    token: str
    logprob: float
    rank: int


class Completion(BaseModel):
    """
    Represents a single generated completion as a sequence of token logprobs.

    Attributes:
        token_logprobs: Mapping from token position to top-K logprobs.
    """

    token_logprobs: dict[int, list[float]]  # Mapping from token position to top-K logprobs


class Result(BaseModel):
    """
    Represents the full data for a single query.

    Attributes:
        query_id: The unique identifier for the query.
        query: The query text.
        expected_answers: List of expected correct answers.
        generated_answers: List of generated answers with metadata.
        token_logprobs: Nested sequence of token log probabilities.
    """

    query_id: str
    query: str
    expected_answers: Sequence[str]
    generated_answers: Sequence[dict[str, str]]
    token_logprobs: Sequence[Sequence[Sequence[float]]]


class Dataset(BaseModel):
    """
    Represents the entire dataset with metadata and results.

    Attributes:
        results: Sequence of result objects.
    """

    results: Sequence[Result]
