"""Typed models for the completion formats the parsers consume, and the readers over them.

The logprob path and the generated text are modelled; `extra="ignore"` drops the rest of
the payload. `from_attributes=True` accepts both a raw mapping and an attribute-style
object.

`read_message` is the way to the generated text: every caller that wants what the model
said goes through the validated envelope rather than indexing a raw payload.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

_ACCEPTS_DICT_OR_OBJECT = ConfigDict(from_attributes=True, extra="ignore")

# The 2xx band, as bounds rather than a status enum: the only question asked of a batch
# line's status is whether it succeeded.
_OK = 200
_REDIRECT = 300


class TopLogprob(BaseModel):
    """One rank of the top-k distribution for a single token."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    logprob: float | None = None


class TokenLogprobs(BaseModel):
    """One generated token, with the top-k alternatives considered at that position."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    logprob: float | None = None
    top_logprobs: list[TopLogprob] = []


class ChatChoiceLogprobs(BaseModel):
    """The `logprobs` block of a chat choice, holding one entry per generated token."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    content: list[TokenLogprobs] = []


class ChatMessage(BaseModel):
    """The assistant message a chat choice carries."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    content: str | None = None


class ChatChoice(BaseModel):
    """One sampled sequence of a chat completion."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    logprobs: ChatChoiceLogprobs | None = None
    # The generated text. Nothing in the scoring path reads it -- a detector scores the
    # distribution, not the words -- but the same file is what a caller labels from, and
    # digging the text back out with `["choices"][0]["message"]["content"]` after the
    # envelope has already been validated is the one step that would stay untyped.
    message: ChatMessage | None = None


class ChatCompletion(BaseModel):
    """`client.chat.completions.create(...)` — one `choice` per sampled sequence."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    choices: list[ChatChoice]


class ResponseContentPart(BaseModel):
    """One content part of an output item, holding its per-token logprobs."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    logprobs: list[TokenLogprobs] = []


class ResponseOutputItem(BaseModel):
    """One sampled sequence of a Responses API payload."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    content: list[ResponseContentPart] = []


class ResponsesPayload(BaseModel):
    """`client.responses.create(...)` — one `output` item per sampled sequence."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    output: list[ResponseOutputItem]


class BatchResponseData(BaseModel):
    """The `response` envelope of one Batch output line.

    The spec fills a failed envelope two ways. Top-level `error` carries non-HTTP failures
    and leaves the envelope empty; a request the server *rejects* leaves top-level `error`
    null and puts an *error object* in `body` under a 4xx or 5xx `status_code`. A body is
    therefore not evidence of a completion, which is why
    `status_code` has to be read rather than merely modelled, and why it carries no
    default: an absent status is not evidence of success either.

    The completion is carried as it arrived rather than narrowed to `ChatCompletion`,
    because a batch line can carry a `ResponsesPayload` just as well. Consumers validate the
    payload for their own purpose; this envelope's job is the envelope.
    """

    model_config = _ACCEPTS_DICT_OR_OBJECT

    status_code: int
    request_id: str | None = None
    body: Any = None


class BatchRequestOutput(BaseModel):
    """One line of an OpenAI Batch output file.

    The Batch API returns JSONL -- one of these per line -- and any OpenAI-compatible
    server writes the same shape. The completion is always nested under `response.body`.

    `custom_id` is the only id that identifies the request: `id` is assigned by the
    provider, as is `response.request_id`, and so is the completion's own `id`. It is the
    key every stage joins on.

    A failed line carries no usable completion, and `completion` reports that by
    returning `None` rather than raising: reading a batch file is where failures are
    counted, not where a run should abort. What a consumer does with one is its own
    decision -- `LogProbParser` refuses it, because it emits one row per line and a
    dropped row would shift every later response against its label.
    """

    model_config = _ACCEPTS_DICT_OR_OBJECT

    id: str | None = None
    custom_id: str = Field(min_length=1)
    # Required, with no default. Every writer emits both keys -- one of them null -- and a
    # payload carrying neither is not a batch line. Requiring them is what says so for a
    # mapping and an attribute-style object alike, rather than letting an unrecognised
    # payload with a `custom_id` validate here and be reported as a request that failed.
    response: BatchResponseData | None
    error: Any

    @property
    def failure(self) -> str | None:
        """Why this line carries no completion, phrased for an error message.

        `None` when it carries one. The `error` repr is truncated because it is provider
        text of no fixed length, and this ends up inside exception messages.
        """
        if self.error is not None:
            return f"error {self.error!r:.200}"
        if self.response is None:
            return "no response envelope"
        if not _OK <= self.response.status_code < _REDIRECT:
            return f"HTTP {self.response.status_code}"
        if self.response.body is None:
            return "an empty response body"
        return None

    @property
    def completion(self) -> Any | None:
        """The completion payload this line carries, or `None` if it carries none.

        `None` covers every way a line can fail, which `failure` names: a non-HTTP error,
        an HTTP status outside 2xx -- where the body is an error object rather than a
        completion -- and a missing envelope or body.
        """
        if self.failure is not None or self.response is None:
            return None
        return self.response.body


def read_message(completion: Any) -> str | None:
    """What the model said, or `None` if the payload carries no text.

    The single reading of `choices[0].message.content`, so no caller has to index a raw
    payload and none has to decide what an unusable one means. `None` covers every way the
    text can be absent -- no completion at all, as a failed batch line yields; an error
    object where the completion belongs; a completion with no choices; a choice whose
    message or content is null.

    Args:
        completion: A chat completion, as a mapping or an attribute-style object. A batch
            line's `completion` is exactly this.

    Returns:
        The assistant's text, unmodified, or `None`.
    """
    try:
        choices = ChatCompletion.model_validate(completion).choices
    except ValidationError:
        return None
    if not choices or choices[0].message is None:
        return None
    return choices[0].message.content
