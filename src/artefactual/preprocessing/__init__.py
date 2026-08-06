"""Parsing of completion responses into the logprob arrays the scorers consume.

`LogProbParser` is the pipeline step; `parse_top_logprobs` and
`parse_sampled_token_logprobs` are its functional form. The per-format extractors in
`openai_parser` are implementation detail — they take validated models, not raw payloads,
and `parse_top_logprobs` is the supported way in.
"""

from artefactual.preprocessing.parser import (
    LogProbParser,
    parse_sampled_token_logprobs,
    parse_top_logprobs,
)

__all__ = [
    "LogProbParser",
    "parse_sampled_token_logprobs",
    "parse_top_logprobs",
]
