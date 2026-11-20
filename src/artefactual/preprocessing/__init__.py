"""Parsing of completion responses into the logprob arrays the scorers consume.

`LogProbParser` is the pipeline step; `parse_top_logprobs` and
`parse_sampled_token_logprobs` are its functional form. The extraction itself is a method
of the payload models: `ChatCompletion` and `ResponsesPayload` carry `top_logprobs()` and
`sampled_logprobs()` (from `LogprobsMixin`, importable from
`artefactual.preprocessing.response_models`), and `LogprobsMixin.parse` turns a raw payload
into one; `ChatCompletion.parse` and `ResponsesPayload.parse` accept their own format only.

`read_batch` reads a Batch output file into `BatchRequestOutput` rows, and
`index_by_custom_id` keys the successful ones by the id every stage joins on.

`read_message` reads what a model said out of a completion, and `Judgment` reads the other
half of a training set: the verdict an LLM-as-a-judge run wrote into a second one, from a
reply, a completion or a whole batch (`read_judgment` is its bare-boolean form).

The payload models are re-exported here as well, since writing a batch file means building
the shape `read_batch` reads.
"""

from artefactual.preprocessing.batch_output import index_by_custom_id, read_batch
from artefactual.preprocessing.judgments import Judgment, read_judgment
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
    ResponsesPayload,
    read_message,
)

__all__ = [
    "BatchRequestOutput",
    "BatchResponseData",
    "ChatChoice",
    "ChatCompletion",
    "ChatMessage",
    "Judgment",
    "LogProbParser",
    "ResponsesPayload",
    "index_by_custom_id",
    "parse_sampled_token_logprobs",
    "parse_top_logprobs",
    "read_batch",
    "read_judgment",
    "read_message",
]
