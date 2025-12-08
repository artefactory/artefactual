"""Artefactual: A library for LLM response calibration and analysis."""

__version__ = "0.1.0"

from artefactual.estimators import (
    TokenScoringLogisticRegression,
    TokenScoringSGD,
)
from artefactual.parsers import (
    OpenAIParsedCompletionV1,
    extract_logprobs,
    parsed_token_logprobs,
    parsed_top_logprobs_tensor,
)
from artefactual.protocols import (
    LogProbValue,
    ParsedCompletionProtocol,
    TokenInfoProtocol,
)
from artefactual.transformers import (
    EntropicContributionTransformer,
    SequenceStatAggregator,
)


def main() -> None:
    """Main entry point for the artefactual CLI."""
    pass


__all__ = [
    "EntropicContributionTransformer",
    "LogProbValue",
    "OpenAIParsedCompletionV1",
    "ParsedCompletionProtocol",
    "SequenceStatAggregator",
    "TokenInfoProtocol",
    "TokenScoringLogisticRegression",
    "TokenScoringSGD",
    "extract_logprobs",
    "parsed_token_logprobs",
    "parsed_top_logprobs_tensor",
]
