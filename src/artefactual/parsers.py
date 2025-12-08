"""Parsers for OpenAI-style LLM responses with logprobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from absl import logging
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Annotated
from beartype.vale import Is

from artefactual.protocols import DictLike, LogProbValue, ModelDumpable, ParsedCompletionProtocol


@dataclass
class OpenAIParsedCompletionV1:
    """Normalized view of a single OpenAI-compatible completion."""

    text: str
    token_infos: list[TokenInfo]
    metadata: dict[str, Any] | None

    @classmethod
    @beartype
    def from_response(cls, response: dict[str, Any] | Any) -> OpenAIParsedCompletionV1:
        return parse_openai_completion(response)


@dataclass
class TokenInfo:
    """Token-level information."""

    token: str
    logprob: float | None
    top_logprobs: dict[str, float] | None


@beartype
def parse_openai_completion(response: dict[str, Any] | Any) -> OpenAIParsedCompletionV1:
    """Parse a raw OpenAI-style completion response into a normalized object."""
    resp = _coerce_dict(response)
    choice = _first_choice(resp)
    if choice is None:
        return OpenAIParsedCompletionV1(text="", token_infos=[], metadata=None)

    text = _parse_text(choice)
    token_infos = _parse_tokens(choice)
    metadata = _parse_metadata(resp, choice)  # type: ignore[call-arg]

    return OpenAIParsedCompletionV1(text=text, token_infos=token_infos, metadata=metadata)


def parsed_token_logprobs(parsed: list[ParsedCompletionProtocol]) -> np.ndarray:
    """Flatten token logprobs (skipping None) into a 1D numpy array."""
    flat: list[float] = []
    for item in parsed:
        for info in item.token_infos:
            if info.logprob is not None:
                flat.append(info.logprob)
    return np.asarray(flat, dtype=np.float32)


@beartype
def parsed_top_logprobs_tensor(
    parsed: list[ParsedCompletionProtocol],
    top_k: Annotated[int, Is[lambda x: x > 0]],
    max_tokens: Annotated[int | None, Is[lambda x: x is None or x >= 0]] = None,
    padding_value: float = 0.0,
) -> np.ndarray:
    """Convert parsed completions into a padded 3D tensor of top-k logprobs.

    Handles the case where fewer than `top_k` logprobs are returned for a token
    by padding with `padding_value`, and truncates if more are provided.
    """
    seq_lengths = [len(item.token_infos) for item in parsed]
    max_len = max_tokens if max_tokens is not None else (max(seq_lengths) if seq_lengths else 0)

    tensor = np.full((len(parsed), max_len, top_k), padding_value, dtype=np.float32)

    for i, item in enumerate(parsed):
        for j, info in enumerate(item.token_infos[:max_len]):
            top_lp = info.top_logprobs
            if top_lp is None:
                # Fallback to single logprob if available
                if info.logprob is None:
                    continue
                values: list[float] = [info.logprob]
            else:
                # top_logprobs may be a dict[str, float]
                values = sorted(top_lp.values(), reverse=True)
            truncated = values[:top_k]
            # pad/truncate to top_k
            vec = np.full(top_k, padding_value, dtype=np.float32)
            vec[: len(truncated)] = np.array(truncated, dtype=np.float32)
            tensor[i, j] = vec

    return tensor


Dictish = dict[str, Any]
CoercibleDict = Dictish | ModelDumpable | DictLike | Iterable[tuple[str, Any]]


@beartype
def _coerce_dict(obj: CoercibleDict) -> dict[str, Any]:
    candidate: dict[str, Any]
    if is_bearable(obj, Dictish):
        candidate = obj  # type: ignore[assignment]
    elif is_bearable(obj, ModelDumpable):
        candidate = obj.model_dump()
    elif is_bearable(obj, DictLike):
        candidate = obj.dict()
    else:
        candidate = dict(obj)

    message = f"Object of type {type(obj)} cannot be coerced to dict"
    if not is_bearable(candidate, Dictish):
        raise TypeError(message)
    return candidate


@beartype
def _first_choice(resp: dict[str, Any]) -> dict[str, Any] | None:
    if "choices" not in resp or not is_bearable(resp["choices"], list[Any]):
        return None
    choices = resp["choices"]
    if not choices:
        return None
    if len(choices) > 1:
        logging.warning(
            "Multiple choices detected; only the first choice will be parsed. "
            "Request n=1 if you do not need multiple completions."
        )
    raw_choice = choices[0]
    return raw_choice if is_bearable(raw_choice, dict[str, Any]) else _coerce_dict(raw_choice)


@beartype
def _parse_text(choice: dict[str, Any]) -> str:
    if "message" in choice and is_bearable(choice["message"], dict[str, Any]):
        message = choice["message"]
        if "content" in message:
            content = message["content"]
            if is_bearable(content, str):
                return content  # type: ignore[return-value]
            if is_bearable(content, list[Any]):
                parts: list[str] = []
                for part in content:  # type: ignore[arg-type]
                    if is_bearable(part, dict[str, Any]) and "text" in part:
                        parts.append(str(part["text"]))
                    elif is_bearable(part, str):
                        parts.append(part)  # type: ignore[arg-type]
                return "".join(parts)
    if "text" in choice and is_bearable(choice["text"], str):
        return choice["text"]  # type: ignore[return-value]
    return ""


@beartype
def _parse_tokens(choice: dict[str, Any]) -> list[TokenInfo]:  # noqa: C901
    infos: list[TokenInfo] = []

    if "message" in choice and is_bearable(choice["message"], dict[str, Any]):
        message = choice["message"]
        if "content" in message and is_bearable(message["content"], list[Any]):
            content = message["content"]
            for part in content:  # type: ignore[arg-type]
                if not is_bearable(part, dict[str, Any]):
                    continue
                if "logprobs" not in part or not is_bearable(part["logprobs"], dict[str, Any]):
                    continue
                lp = part["logprobs"]
                part_tokens = (
                    lp["tokens"] if "tokens" in lp and is_bearable(lp["tokens"], list[str]) else []
                )
                part_token_logprobs = (
                    lp["token_logprobs"]
                    if "token_logprobs" in lp and is_bearable(lp["token_logprobs"], list[float])
                    else []
                )
                part_top_logprobs = (
                    lp["top_logprobs"]
                    if "top_logprobs" in lp and is_bearable(lp["top_logprobs"], list[dict[str, float]])
                    else []
                )
                n = min(len(part_tokens), len(part_token_logprobs))
                for idx in range(n):
                    alt = part_top_logprobs[idx] if part_top_logprobs else None
                    infos.append(TokenInfo(token=part_tokens[idx], logprob=part_token_logprobs[idx], top_logprobs=alt))

    if not infos and "logprobs" in choice and is_bearable(choice["logprobs"], dict[str, Any]):
        lp = choice["logprobs"]
        tokens = lp["tokens"] if "tokens" in lp and is_bearable(lp["tokens"], list[str]) else []
        token_logprobs = (
            lp["token_logprobs"]
            if "token_logprobs" in lp and is_bearable(lp["token_logprobs"], list[float])
            else []
        )
        if "top_logprobs" in lp and is_bearable(lp["top_logprobs"], list[dict[str, float]]):
            top_logprobs: list[dict[str, float]] | None = lp["top_logprobs"]
        else:
            top_logprobs = None
        if tokens and token_logprobs:
            n = min(len(tokens), len(token_logprobs))
            token_list = list(tokens[:n])
            logprob_list = list(token_logprobs[:n])
            top_logprob_list = list(top_logprobs[:n]) if top_logprobs else [None] * n
            for i in range(n):
                infos.append(TokenInfo(token=token_list[i], logprob=logprob_list[i], top_logprobs=top_logprob_list[i]))

    return infos


@beartype
def extract_logprobs(logprobs: list[dict[int, LogProbValue]]) -> tuple[list[int], list[float]]:
    """Extract token ids and logprob values from a list of dicts."""
    tokens: list[int] = []
    lps: list[float] = []
    for mapping in logprobs:
        for tok, lp in mapping.items():
            tokens.append(tok)
            lps.append(lp.logprob)
    return tokens, lps


@beartype
def _parse_metadata(resp: dict[str, Any], choice: dict[str, Any]) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    for key in ("id", "model", "usage", "created", "object"):
        if key in resp:
            metadata[key] = resp[key]
    if "finish_reason" in choice:
        metadata["finish_reason"] = choice["finish_reason"]
    return metadata or None
