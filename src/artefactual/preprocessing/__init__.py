"""Parsing of completion responses into the logprob arrays the scorers consume.

`LogProbParser` is the pipeline step; `parse_top_logprobs` and
`parse_sampled_token_logprobs` are its functional form. The per-format extractors in
`openai_parser` are implementation detail — they take validated models, not raw payloads,
and `parse_top_logprobs` is the supported way in.

`read_batch` reads a Batch output file into `BatchRequestOutput` rows, and
`index_by_custom_id` keys the successful ones by the id every stage joins on.

`read_message` reads what a model said out of a completion, and `read_judgment` reads the
other half of a training set: the verdict an LLM-as-a-judge run wrote into a second one.

The payload models are re-exported here as well, since writing a batch file means building
the shape `read_batch` reads.
"""

from artefactual.preprocessing.batch_output import index_by_custom_id, read_batch
from artefactual.preprocessing.judgments import read_judgment
from artefactual.preprocessing.parser import (
    LogProbParser,
    parse_sampled_token_logprobs,
    parse_top_logprobs,
)
from artefactual.preprocessing.response_models import (
    BatchRequestOutput,
    BatchResponseData,
    ChatChoice,
    ChatCompletion,
    ChatMessage,
    read_message,
)

__all__ = [
    "BatchRequestOutput",
    "BatchResponseData",
    "ChatChoice",
    "ChatCompletion",
    "ChatMessage",
    "LogProbParser",
    "index_by_custom_id",
    "parse_sampled_token_logprobs",
    "parse_top_logprobs",
    "read_batch",
    "read_judgment",
    "read_message",
]
