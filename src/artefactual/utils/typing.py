"""Type-checking configuration and validators shared by the estimators that validate user input."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from beartype import BeartypeConf
from beartype.door import is_bearable
from beartype.vale import Is
from pydantic import BaseModel

TYPE_CHECKED = BeartypeConf(violation_param_type=TypeError)


def is_logprob_array(X: Any) -> bool:  # ruff: ignore[invalid-argument-name]
    """Whether `X` is not a completion response, which only a `LogProbParser` turns into an array.

    A response object or mapping, or a non-empty sequence of them, is not a logprob array.
    """
    single = is_bearable(X, BaseModel | Mapping[str, Any])  # a scipy `dok_matrix` is a Mapping keyed by tuples
    return not (single or (is_bearable(X, Sequence[BaseModel | Mapping]) and len(X)))


LogprobArray = Annotated[Any, Is[is_logprob_array]]
