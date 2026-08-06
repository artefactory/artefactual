"""Format-specific extractors for the two OpenAI completion shapes.

`chat.completions` nests one sequence per `choice`; `responses` nests one per `output`
item. Both are handed a model from `response_models` that `parser` has already validated,
so fields are read by attribute: the "mapping or object?" question the payload arrives
with is settled at that boundary, and a raw mapping never reaches this module.

Dispatch between them lives in `parser`; nothing here is called directly by the pipeline.
"""

import numpy as np

from artefactual.preprocessing.response_models import (
    ChatCompletion,
    ResponsesPayload,
    TokenLogprobs,
)


def _ranks(token: TokenLogprobs) -> list[float]:
    """The token's top-k logprobs, highest first; empty if the position carried none.

    Sorted here rather than left to the transformer because `LogProbParser` truncates to
    `k` ranks, and truncating an unordered list would keep the wrong ones.
    """
    return sorted((rank.logprob for rank in token.top_logprobs if rank.logprob is not None), reverse=True)


def process_openai_chat_completion(response: ChatCompletion) -> list[dict[int, list[float]]]:
    """Extract per-token top logprobs from a chat completion, one dict per choice.

    Args:
        response: A validated `ChatCompletion`.

    Returns:
        One dict per sampled sequence, mapping token position to that token's ranks.
        Positions whose `top_logprobs` was empty are omitted.
    """
    sequences = []
    for choice in response.choices:
        sequence: dict[int, list[float]] = {}
        content = choice.logprobs.content if choice.logprobs is not None else []
        for token_index, token in enumerate(content):
            ranks = _ranks(token)
            if ranks:
                sequence[token_index] = ranks
        sequences.append(sequence)
    return sequences


def process_openai_responses_api(response: ResponsesPayload) -> list[dict[int, list[float]]]:
    """Extract per-token top logprobs from a Responses payload, one dict per output item.

    Structure: `response.output -> [item] -> item.content -> [part] -> part.logprobs`.

    Args:
        response: A validated `ResponsesPayload`.

    Returns:
        One dict per sampled sequence, mapping token position to that token's ranks.
        Positions whose `top_logprobs` was empty are omitted.
    """
    batch = []
    for item in response.output:
        sequence: dict[int, list[float]] = {}
        # The index is the token's position in the generated sequence, so it advances for
        # every token -- including one whose top_logprobs came back empty. Skipping the
        # increment there would shift every later token onto the wrong word.
        token_index = 0
        for part in item.content:
            for token in part.logprobs:
                # Unlike the chat format, a Responses token may carry only the sampled
                # logprob. Falling back to it keeps the position from being dropped.
                ranks = _ranks(token) if token.top_logprobs else _sampled_only(token)
                if ranks:
                    sequence[token_index] = ranks
                token_index += 1
        batch.append(sequence)
    return batch


def _sampled_only(token: TokenLogprobs) -> list[float]:
    """The sampled token's own logprob as a one-element rank list, or empty if absent."""
    return [token.logprob] if token.logprob is not None else []


def sampled_tokens_logprobs_responses_api(response: ResponsesPayload) -> list[np.ndarray]:
    """Logprobs of the sampled tokens only, one 1-D array per output item.

    Args:
        response: A validated `ResponsesPayload`.

    Returns:
        One array per sampled sequence, holding the sampled token logprobs in order.
    """
    return [
        np.array([token.logprob for part in item.content for token in part.logprobs if token.logprob is not None])
        for item in response.output
    ]


def sampled_tokens_logprobs_chat_completion_api(response: ChatCompletion) -> list[np.ndarray]:
    """Logprobs of the sampled tokens only, one 1-D array per choice.

    Args:
        response: A validated `ChatCompletion`.

    Returns:
        One array per sampled sequence, holding the sampled token logprobs in order.
    """
    return [
        np.array([
            token.logprob
            for token in (choice.logprobs.content if choice.logprobs is not None else [])
            if token.logprob is not None
        ])
        for choice in response.choices
    ]
