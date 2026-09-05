"""Typed models for the completion formats the parsers consume.

Only the logprob path is modelled; `extra="ignore"` drops the rest of the payload.
`from_attributes=True` accepts both a raw mapping and an attribute-style object.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class BatchResponseData(BaseModel):
    """The `response` envelope of one Batch output line.

    A failed request fills this envelope differently depending on who wrote the file.
    `vllm run-batch` leaves `body` out and reports the reason in the line's top-level
    `error`; the OpenAI Batch API leaves top-level `error` null -- it documents that field
    as carrying non-HTTP failures only -- and puts an *error object* in `body` under a 4xx
    or 5xx `status_code`. A body is therefore not evidence of a completion, which is why
    `status_code` has to be read rather than merely modelled.

    The completion is carried as it arrived rather than narrowed to `ChatCompletion`.
    That model covers the logprob path only -- by design, since that is all the detector
    reads -- so validating into it here would discard `message.content`, which is where an
    LLM-as-a-judge verdict lives. Consumers validate the payload for their own purpose;
    this envelope's job is the envelope.
    """

    model_config = _ACCEPTS_DICT_OR_OBJECT

    status_code: int = 200
    request_id: str | None = None
    body: Any = None


def _is_bare_completion(response: Any) -> bool:
    """Whether `response` is a payload put straight where the envelope belongs.

    Told apart by what a payload has and an envelope does not: `choices` on a chat
    completion, `output` on a Responses payload. The envelope's own marker, `body`, is
    checked first so an envelope carrying either key inside is never mistaken for one.
    """
    if response is None:
        return False
    if isinstance(response, dict):
        return "body" not in response and ("choices" in response or "output" in response)
    return not hasattr(response, "body") and (hasattr(response, "choices") or hasattr(response, "output"))


class BatchRequestOutput(BaseModel):
    """One line of an OpenAI Batch output file.

    The Batch API returns JSONL -- one of these per line -- and this is the shape
    `vllm run-batch` writes. Neither the OpenAI SDK nor this package modelled it, so every
    reader unwrapped it by hand: `scripts/train_detector.py` in Python and
    `scripts/ecir/build_judge_requests.sh` in jq, each with its own copy of the same
    `(.response.body // .response)` fallback.

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
    response: BatchResponseData | None = None
    error: Any = None

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_completion(cls, data: Any) -> Any:
        """Wrap a completion that was put directly in `response`, as older vllm did.

        The Batch spec nests it under `body`; versions before that emitted it bare. Both
        are still in the wild, so the older shape is normalised here rather than left for
        every caller to sniff.

        The attribute-style branch is not symmetry for its own sake: an in-process
        `vllm.entrypoints.openai.protocol.BatchRequestOutput` is an object, not a mapping,
        and without it the line would validate with no body and be reported as a request
        that failed -- a silent misread rather than an error.
        """
        if isinstance(data, dict):
            # A mapping with a `custom_id` and neither of the two keys that say how the
            # request went is not a batch line; without this it validates into one, and an
            # unrecognised payload that happens to carry that key is reported as a batch
            # line whose request failed rather than as an unsupported format.
            if not ({"response", "error"} & data.keys()):
                message = "a Batch output line carries `response` or `error`"
                raise ValueError(message)
            if _is_bare_completion(data.get("response")):
                return {**data, "response": {"body": data["response"]}}
            return data
        if _is_bare_completion(getattr(data, "response", None)):
            return {
                "id": getattr(data, "id", None),
                "custom_id": getattr(data, "custom_id", None),
                "response": {"body": data.response},
                "error": getattr(data, "error", None),
            }
        return data

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
