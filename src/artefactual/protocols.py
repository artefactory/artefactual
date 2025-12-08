"""Shared typing protocols for artefactual components.

These Protocols enable static typing and runtime validation (via beartype) for
objects passed between parsers, feature builders, and estimators.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LogProbValue(Protocol):
    """Protocol for objects that expose a `logprob` float attribute."""

    logprob: float


@runtime_checkable
class TokenInfoProtocol(Protocol):
    """Protocol for token-level structures emitted by parsers."""

    token: str
    logprob: float | None
    top_logprobs: dict[str, float] | None


@runtime_checkable
class ParsedCompletionProtocol(Protocol):
    """Protocol for parsed completion objects."""

    text: str
    token_infos: list[TokenInfoProtocol]
    metadata: dict[str, Any] | None


@runtime_checkable
class ModelDumpable(Protocol):
    """Protocol for objects exposing a Pydantic-style model_dump()."""

    def model_dump(self) -> dict[str, Any]:
        ...


@runtime_checkable
class DictLike(Protocol):
    """Protocol for objects exposing a dict() method."""

    def dict(self) -> dict[str, Any]:
        ...
