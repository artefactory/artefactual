"""Typed models for the completion formats the parsers consume, and the readers over them.

The logprob path and the generated text are modelled; `extra="ignore"` drops the rest of
the payload. `from_attributes=True` accepts both a raw mapping and an attribute-style
object.

`read_message` is the way to the generated text: every caller that wants what the model
said goes through the validated envelope rather than indexing a raw payload.

The two completion formats carry the logprob extraction themselves, through `LogprobsMixin`:
a format supplies its token `sequences()` and inherits `top_logprobs()` and
`sampled_logprobs()`. `LogprobsMixin.parse` turns whatever a caller holds into the format it
is, and `ChatCompletion.parse` or `ResponsesPayload.parse` accepts that format only. A new
format is one more model implementing the mixin: it registers itself, so no dispatch table
is edited.
"""

import functools
import operator
from http import HTTPStatus
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class ResponseModel(BaseModel):
    """Base of every payload model: accepts a mapping or an attribute-style object, and
    ignores the keys it does not model."""

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class TopLogprob(ResponseModel):
    """One candidate of the top-k distribution for a single token.

    Its rank is its position in that distribution, counting from the most likely.
    """

    logprob: float | None = None


class TokenLogprobs(ResponseModel):
    """One generated token, with the top-k alternatives considered at that position."""

    logprob: float | None = None
    top_logprobs: list[TopLogprob] = []

    def candidates(self, *, fallback_to_sampled: bool = False) -> list[float]:
        """The token's top-k logprobs, highest first; empty if the position carried none.

        With `fallback_to_sampled`, a token whose `top_logprobs` is an empty list is its
        own sole candidate (empty if its logprob is null too). A token whose candidates
        are all null gets no fallback: it did carry a distribution.

        Sorted here rather than left to the transformer because `LogProbParser` truncates
        to the `k` most likely candidates, and truncating an unordered list would keep the
        wrong ones. A candidate whose logprob is null is dropped rather than coerced.
        """
        if fallback_to_sampled and not self.top_logprobs:
            return [] if self.logprob is None else [self.logprob]
        return sorted(
            (candidate.logprob for candidate in self.top_logprobs if candidate.logprob is not None), reverse=True
        )


class LogprobsMixin:
    """Logprob extraction for a completion format that supplies its token `sequences()`.

    Mixed into `ChatCompletion` and `ResponsesPayload`; it holds no state and has no
    `__init__`. Everything it offers is written once over `sequences()`, so the formats
    differ only in where they keep their tokens. A pydantic model that inherits it is
    registered in `formats`, which `LogprobsMixin.parse` matches a payload against.

    `parse` is the way in from raw input: on the mixin it accepts a chat completion, a
    Responses payload or one line of a Batch output file, as a mapping or an
    attribute-style object; on a format it accepts that format only.
    """

    # Every pydantic model that inherits the mixin directly, in definition order, which is
    # the order union matching tries them in.
    formats: ClassVar[list[type]] = []
    # Whether a token carrying no `top_logprobs` is its own sole candidate. A Responses token
    # may carry only the sampled logprob; a chat token never does.
    sampled_fallback: ClassVar[bool] = False
    # Validators by (class, with a batch line admitted), built on first use because the
    # models they name are not all defined when the mixin is, and dropped when a format
    # registers.
    adapters: ClassVar[dict[tuple[type, bool], TypeAdapter[Any]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register a direct implementer as a format.

        Only a direct one: a subclass of a format is that format, and listing both would make
        union matching ambiguous.
        """
        super().__init_subclass__(**kwargs)
        if LogprobsMixin in cls.__bases__:
            LogprobsMixin.formats.append(cls)
            LogprobsMixin.adapters.clear()

    def sequences(self) -> "list[list[TokenLogprobs]]":
        """The generated tokens of every sampled sequence, one list per sequence.

        A sequence without logprobs is `[]`. A token position is its index in the list, so
        a token carrying no candidates still takes its place.
        """
        raise NotImplementedError

    def top_logprobs(self) -> list[dict[int, list[float]]]:
        """Each token's top-k logprobs, one dict per sampled sequence.

        Returns
        -------
        list of dict
            Maps token position to that token's candidate log-probabilities, most likely
            first. Positions whose `top_logprobs` was empty are omitted, and the index is
            still counted so positions track the generated text. When `sampled_fallback`
            is set, such a position holds its sampled logprob alone instead.
        """
        return [
            {
                position: candidates
                for position, token in enumerate(tokens)
                if (candidates := token.candidates(fallback_to_sampled=self.sampled_fallback))
            }
            for tokens in self.sequences()
        ]

    def sampled_logprobs(self) -> list[np.ndarray]:
        """The logprob of each sampled token, ignoring the alternatives.

        Neither EPR nor WEPR uses this; it is for callers measuring sequence likelihood.

        Returns
        -------
        list of ndarray
            One 1-D array per sampled sequence, holding the sampled token logprobs in
            order. A token whose logprob is null is skipped.
        """
        return [
            np.array([token.logprob for token in tokens if token.logprob is not None]) for tokens in self.sequences()
        ]

    @classmethod
    def adapter(cls, *, batch: bool = True) -> TypeAdapter[Any]:
        """The validator for the formats `cls` stands for: every registered one on the
        mixin, `cls` alone on a format.

        Parameters
        ----------
        batch : bool, default=True
            Also accept a `BatchRequestOutput`, which `parse` then opens. A batch line's own
            body is validated without it: a body is a completion, and admitting a line
            would let a body with a `custom_id` validate as an envelope and be unwrapped
            again.
        """
        key = (cls, batch)
        if key not in LogprobsMixin.adapters:
            members = LogprobsMixin.formats if cls is LogprobsMixin else [cls]
            union = functools.reduce(operator.or_, [*members, BatchRequestOutput] if batch else members)
            LogprobsMixin.adapters[key] = TypeAdapter(union)
        return LogprobsMixin.adapters[key]

    @classmethod
    def parse(cls, outputs: Any) -> "ChatCompletion | ResponsesPayload":
        """Validate a payload against the completion formats, telling the two failures apart.

        Called on `LogprobsMixin` it matches every registered format; called on a format it
        matches that one, so `ChatCompletion.parse` refuses a Responses payload.

        A payload that matches no format is not a completion response, which is a
        `TypeError` about the input's kind. A payload that *is* recognisably one but
        carries bad data is a validation failure, and pydantic's own report names the
        offending field; flattening it into "unsupported format" would send the reader
        looking for an unsupported provider instead.

        A union member counts as recognised only once it reaches *inside* the structure a
        format describes. A top-level key that is absent, or present carrying the wrong
        kind of value, says nothing about the payload's intent -- an arbitrary mapping that
        happens to hold a key named `output` is not a Responses payload with one bad field.

        Parameters
        ----------
        outputs : Any
            A single completion response, as a mapping or an object; a batch line is opened
            and its body returned.

        Returns
        -------
        ChatCompletion or ResponsesPayload
            The format the payload is; with `cls` a format, an instance of `cls`.

        Raises
        ------
        TypeError
            If the payload matches no format `cls` stands for, including a batch line whose
            body matches none.
        ValidationError
            If it matches one and a field inside it is invalid.
        ValueError
            If a batch line carries no completion, because its request failed.
        """
        try:
            response = cls.adapter().validate_python(outputs, from_attributes=True)
        except ValidationError as error:
            # Errors are located as `(member, field, ...)`. Depth one is the member refusing
            # the input's type outright; depth two is a top-level key absent or holding the
            # wrong kind of value, which any unrelated mapping can produce by coincidence.
            # Only a failure below that names a payload a member walked into and objected to.
            if any(len(item["loc"]) > 2 for item in error.errors()):
                raise
            msg = (
                f"Unsupported output format: {type(outputs).__name__}. Expected a completion response carrying "
                "logprobs. A numeric array of logprobs is scored by EPRClassifier or WEPRClassifier, not by "
                "this parser or the detectors."
            )
            raise TypeError(msg) from error
        match response:
            case BatchRequestOutput():
                return response.payload(cls)
            case _:
                return response


class ChatChoiceLogprobs(ResponseModel):
    """The `logprobs` block of a chat choice, holding one entry per generated token."""

    content: list[TokenLogprobs] = []


class ChatMessage(ResponseModel):
    """The assistant message a chat choice carries."""

    content: str | None = None


class ChatChoice(ResponseModel):
    """One sampled sequence of a chat completion."""

    logprobs: ChatChoiceLogprobs | None = None
    # The generated text. Nothing in the scoring path reads it -- a detector scores the
    # distribution, not the words -- but the same file is what a caller labels from, and
    # digging the text back out with `["choices"][0]["message"]["content"]` after the
    # envelope has already been validated is the one step that would stay untyped.
    message: ChatMessage | None = None


class ChatCompletion(LogprobsMixin, ResponseModel):
    """`client.chat.completions.create(...)` — one `choice` per sampled sequence.

    Extraction comes from `LogprobsMixin`: `top_logprobs()`, `sampled_logprobs()` and the
    classmethod `parse`, which accepts this format, bare or in a batch line, only.
    """

    choices: list[ChatChoice]

    def sequences(self) -> list[list[TokenLogprobs]]:
        """One token list per choice; a choice without logprobs is `[]`."""
        return [choice.logprobs.content if choice.logprobs is not None else [] for choice in self.choices]


class ResponseContentPart(ResponseModel):
    """One content part of an output item, holding its per-token logprobs."""

    logprobs: list[TokenLogprobs] = []


class ResponseOutputItem(ResponseModel):
    """One sampled sequence of a Responses API payload."""

    content: list[ResponseContentPart] = []


class ResponsesPayload(LogprobsMixin, ResponseModel):
    """`client.responses.create(...)` — one `output` item per sampled sequence.

    Extraction comes from `LogprobsMixin`: `top_logprobs()`, `sampled_logprobs()` and the
    classmethod `parse`, which accepts this format, bare or in a batch line, only.
    """

    sampled_fallback: ClassVar[bool] = True

    output: list[ResponseOutputItem]

    def sequences(self) -> list[list[TokenLogprobs]]:
        """One token list per output item, its content parts concatenated in order.

        The tokens are the validated ones. A Responses token may carry only the sampled
        logprob, which `sampled_fallback` makes its own sole candidate.
        """
        return [[token for part in item.content for token in part.logprobs] for item in self.output]


class BatchResponseData(ResponseModel):
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

    status_code: int
    request_id: str | None = None
    body: Any = None


class BatchRequestOutput(ResponseModel):
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
        # The 2xx band: the only question asked of a batch line's status is whether it succeeded.
        if not HTTPStatus.OK <= self.response.status_code < HTTPStatus.MULTIPLE_CHOICES:
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

    def payload(self, kind: type[LogprobsMixin] = LogprobsMixin) -> ChatCompletion | ResponsesPayload:
        """The completion this line carries, validated as the format it is.

        `kind` narrows the formats accepted: the default is every registered one, a format
        accepts its own only.

        Refused rather than skipped for a line that carries none. A step that emits one row
        per sampled sequence would shift every later response against its label by
        dropping a failed line -- a silently mistrained detector rather than an error.
        Filter the failed lines out before parsing, which is where their failure can still
        be reported against a `custom_id`.

        The body goes through the same formats as a bare completion, so a batch of
        Responses-API calls parses like any other.

        Raises
        ------
        ValueError
            If the line failed, naming its `custom_id` and `failure`.
        TypeError
            If the body is neither a chat completion nor a Responses payload.
        """
        if self.completion is None:
            msg = (
                f"Batch line {self.custom_id!r} carries no completion: {self.failure}. "
                f"Drop failed lines before parsing -- keeping them would pair every later "
                f"response with the wrong label."
            )
            raise ValueError(msg)
        try:
            return kind.adapter(batch=False).validate_python(self.completion)
        except ValidationError as error:
            what = (
                "neither a chat completion nor a Responses payload"
                if kind is LogprobsMixin
                else f"not a {kind.__name__}"
            )
            msg = f"Batch line {self.custom_id!r} carries a body that is {what}."
            raise TypeError(msg) from error


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
