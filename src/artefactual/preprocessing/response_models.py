"""Typed models for the completion formats the parsers consume.

Only the logprob path is modelled; `extra="ignore"` drops the rest of the payload.
`from_attributes=True` accepts both a raw mapping and an attribute-style object.
"""

from pydantic import BaseModel, ConfigDict

_ACCEPTS_DICT_OR_OBJECT = ConfigDict(from_attributes=True, extra="ignore")


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


class ChatChoice(BaseModel):
    """One sampled sequence of a chat completion."""

    model_config = _ACCEPTS_DICT_OR_OBJECT

    logprobs: ChatChoiceLogprobs | None = None


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
