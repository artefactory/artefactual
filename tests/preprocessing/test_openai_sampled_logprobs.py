"""
Behavioral tests for sampled-token log-probability extraction from OpenAI API responses.

These tests verify the *contract* between the OpenAI API response format and the
library's extraction functions — independently of the implementation details.

The two functions under test are:
  - sampled_tokens_logprobs_chat_completion_api
  - sampled_tokens_logprobs_responses_api

Both API formats are documented at:
  https://platform.openai.com/docs/api-reference/chat/object
  https://platform.openai.com/docs/api-reference/responses

Key contract properties verified here:
  1. Sampled logprob identity  — returns `logprob` (the generated token), not
     any value from `top_logprobs` (the top-K alternatives).
  2. Value fidelity           — parsed values exactly match the API response.
  3. Value range              — all logprobs are ≤ 0 and finite (log-probability
                               axiom: log(p) ≤ 0 for p ∈ (0, 1]).
  4. Cardinality              — one result per choice / output item.
  5. Length fidelity          — each result entry has one value per token.
  6. Variable-length batches  — ragged batches (normal in production) do not crash.
  7. Empty responses          — empty choices / output lists return empty results.
  8. Cross-format consistency — both parsers agree on numerically equivalent data.
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from artefactual.preprocessing.openai_parser import (
    sampled_tokens_logprobs_chat_completion_api,
    sampled_tokens_logprobs_responses_api,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

logprob_st = st.floats(min_value=-50.0, max_value=-1e-6, allow_nan=False, allow_infinity=False)
nonempty_seq_st = st.lists(logprob_st, min_size=1, max_size=30)

# uniform: all sequences have the same token count — for correctness tests that
# must not be confounded by the ragged-array crash at openai_parser.py:209/240.
uniform_sequences_st = st.integers(min_value=1, max_value=6).flatmap(
    lambda n_seqs: st.integers(min_value=1, max_value=20).flatmap(
        lambda seq_len: st.lists(
            st.lists(logprob_st, min_size=seq_len, max_size=seq_len),
            min_size=n_seqs,
            max_size=n_seqs,
        )
    )
)

# ragged: at least two sequences with different lengths — for variable-length tests.
ragged_sequences_st = st.lists(nonempty_seq_st, min_size=2, max_size=6).filter(
    lambda lol: len({len(seq) for seq in lol}) > 1
)


# ---------------------------------------------------------------------------
# Realistic API response builders
#
# These builders mirror the actual OpenAI API response structure so that the
# tests document the *expected API shape*, not just a convenient internal fixture.
#
# ChatCompletion logprobs structure:
#   choices[i].logprobs.content[j].logprob        ← sampled token logprob
#   choices[i].logprobs.content[j].top_logprobs[] ← top-K alternatives
#
# Responses API logprobs structure:
#   output[i].content[k].logprobs[j].logprob        ← sampled token logprob
#   output[i].content[k].logprobs[j].top_logprobs[] ← top-K alternatives
# ---------------------------------------------------------------------------


def _chat_completion_response(
    logprob_lists: list[list[float]],
    *,
    include_top_logprobs: bool = False,
    top_logprob_offset: float = 1.0,
) -> dict:
    """
    Build a realistic ChatCompletion response dict with per-token logprobs.

    When `include_top_logprobs=True`, each token entry also contains a
    `top_logprobs` field whose values are shifted by `top_logprob_offset`
    from the sampled logprob, so tests can distinguish which field was used.
    """
    choices = []
    for lps in logprob_lists:
        content = []
        for lp in lps:
            token_entry: dict = {"token": "<tok>", "logprob": lp, "bytes": []}
            if include_top_logprobs:
                token_entry["top_logprobs"] = [
                    {"token": "<top1>", "logprob": min(lp + top_logprob_offset, -1e-9), "bytes": []}, # Ensure top_logprob is negative and distinct
                    {"token": "<top2>", "logprob": lp - 1.0, "bytes": []},
                ]
            content.append(token_entry)
        choices.append({
            "index": len(choices),
            "message": {"role": "assistant", "content": ""},
            "logprobs": {"content": content},
            "finish_reason": "stop",
        })
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": choices,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _responses_api_response(
    logprob_lists: list[list[float]],
    *,
    include_top_logprobs: bool = False,
    top_logprob_offset: float = 1.0,
) -> dict:
    """
    Build a realistic Responses API response dict with per-token logprobs.

    Structure follows the OpenAI Responses API:
      response.output[i].content[k].logprobs[j].logprob
    """
    output = []
    for lps in logprob_lists:
        token_logprobs = []
        for lp in lps:
            entry: dict = {"token": "<tok>", "logprob": lp, "bytes": []}
            if include_top_logprobs:
                entry["top_logprobs"] = [
                    {"token": "<top1>", "logprob": min(lp + top_logprob_offset, -1e-9), "bytes": []}, # Ensure top_logprob is negative and distinct
                    {"token": "<top2>", "logprob": lp - 1.0, "bytes": []},
                ]
            token_logprobs.append(entry)
        output.append({
            "id": f"msg_{len(output)}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "", "logprobs": token_logprobs}],
            "status": "completed",
        })
    return {
        "id": "resp_test",
        "object": "response",
        "model": "gpt-4o",
        "output": output,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


# ---------------------------------------------------------------------------
# Fixture: both API parsers share the same contract, so each property is
# tested once with both (parser_func, response_builder) via this fixture.
# ---------------------------------------------------------------------------


@pytest.fixture(params=[
    (sampled_tokens_logprobs_chat_completion_api, _chat_completion_response),
    (sampled_tokens_logprobs_responses_api, _responses_api_response),
], ids=["chat_completion", "responses_api"])
def parser_pair(request):
    return request.param


# ---------------------------------------------------------------------------
# 1. Sampled logprob identity — `logprob`, not `top_logprobs`
# ---------------------------------------------------------------------------


@given(logprob_lists=uniform_sequences_st)
def test_returns_sampled_logprob_not_top_k(parser_pair, logprob_lists):
    """
    The parser must extract the sampled token's logprob, not any value from
    top_logprobs. The top_logprobs values are shifted by +1.0 so the two
    fields are distinguishable regardless of their relative magnitude.
    """
    parser_func, response_builder = parser_pair
    response = response_builder(logprob_lists, include_top_logprobs=True)
    result = parser_func(response)
    for seq_result, expected in zip(result, logprob_lists):
        np.testing.assert_allclose(seq_result, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. Value range (log-probability axiom: logprob ≤ 0)
# ---------------------------------------------------------------------------


@given(logprob_lists=uniform_sequences_st)
def test_all_logprobs_non_positive(parser_pair, logprob_lists):
    """All extracted logprobs must satisfy logprob ≤ 0."""
    parser_func, response_builder = parser_pair
    response = response_builder(logprob_lists)
    result = parser_func(response)
    for seq in result:
        assert np.all(np.array(seq) <= 0.0)


@given(logprob_lists=uniform_sequences_st)
def test_all_logprobs_finite(parser_pair, logprob_lists):
    """Extracted logprobs must be finite — no NaN, no ±inf."""
    parser_func, response_builder = parser_pair
    response = response_builder(logprob_lists)
    result = parser_func(response)
    for seq in result:
        assert np.all(np.isfinite(seq))


# ---------------------------------------------------------------------------
# 4 & 5. Cardinality and length fidelity
# ---------------------------------------------------------------------------


@given(logprob_lists=uniform_sequences_st)
def test_one_result_per_sequence(parser_pair, logprob_lists):
    """One result entry per choice / output item in the API response."""
    parser_func, response_builder = parser_pair
    response = response_builder(logprob_lists)
    result = parser_func(response)
    assert len(result) == len(logprob_lists)


@given(logprob_lists=uniform_sequences_st)
def test_result_length_matches_token_count(parser_pair, logprob_lists):
    """Each result entry must contain one logprob per generated token."""
    parser_func, response_builder = parser_pair
    response = response_builder(logprob_lists)
    result = parser_func(response)
    for seq_result, expected in zip(result, logprob_lists):
        assert len(seq_result) == len(expected)


# ---------------------------------------------------------------------------
# 6. Variable-length batches (ragged inputs — the normal production case)
#
# Different prompts produce responses of different token counts. The parsers
# must handle ragged batches without crashing. np.array() on ragged lists
# raises ValueError in NumPy 2 — see openai_parser.py:209 and :240.
# ---------------------------------------------------------------------------


@given(logprob_lists=ragged_sequences_st)
def test_ragged_batch_does_not_crash(parser_pair, logprob_lists):
    """
    Sequences with different token counts (ragged batch) must not crash.
    The parser must return one result entry per sequence, with correct values.
    """
    parser_func, response_builder = parser_pair
    response = response_builder(logprob_lists)
    result = parser_func(response)
    assert len(result) == len(logprob_lists)
    for seq_result, expected in zip(result, logprob_lists):
        np.testing.assert_allclose(seq_result, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# 7. Empty responses
# ---------------------------------------------------------------------------


def test_chat_completion_empty_choices_returns_empty():
    """An API response with zero choices must return an empty result."""
    response = {"id": "chatcmpl-test", "object": "chat.completion", "choices": []}
    result = sampled_tokens_logprobs_chat_completion_api(response)
    assert len(result) == 0


def test_responses_api_empty_output_returns_empty():
    """An API response with zero output items must return an empty result."""
    response = {"id": "resp_test", "object": "response", "output": []}
    result = sampled_tokens_logprobs_responses_api(response)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 8. Cross-format consistency
#
# Both parsers expose the same semantic contract. Given structurally equivalent
# logprob data, they must return numerically identical results. This prevents
# silent divergence between the two API format handlers as either the API or
# the library evolves.
# ---------------------------------------------------------------------------


@given(uniform_sequences_st)
def test_both_parsers_agree_on_equivalent_logprob_data(logprob_lists):
    """
    ChatCompletion and Responses API parsers must return numerically identical
    results when given the same underlying logprob values in their respective
    API formats.
    """
    chat_result = sampled_tokens_logprobs_chat_completion_api(
        _chat_completion_response(logprob_lists)
    )
    resp_result = sampled_tokens_logprobs_responses_api(
        _responses_api_response(logprob_lists)
    )
    assert len(chat_result) == len(resp_result)
    for chat_seq, resp_seq in zip(chat_result, resp_result):
        np.testing.assert_allclose(chat_seq, resp_seq, rtol=1e-6)
