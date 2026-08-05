"""Edge cases in the OpenAI extraction helpers.

`test_openai_parsing.py` covers the happy path over generated payloads. This file covers
the degenerate shapes real providers emit — absent `logprobs`, a token whose `top_logprobs`
came back empty, a `logprob` of `None` — and the places where the two wire formats are
supposed to mean the same thing.
"""

import numpy as np
import pytest

from artefactual.preprocessing.openai_parser import (
    _extract_logprobs_from_token,
    _get_val,
    _parse_token_entry,
    is_openai_responses_api,
    process_openai_chat_completion,
    process_openai_responses_api,
    sampled_tokens_logprobs_chat_completion_api,
    sampled_tokens_logprobs_responses_api,
)


def chat(tokens, n_choices=1):
    return {"choices": [{"logprobs": {"content": tokens}} for _ in range(n_choices)]}


def responses(tokens):
    return {"output": [{"content": [{"logprobs": tokens}]}]}


def token(*logprobs):
    return {"top_logprobs": [{"logprob": value} for value in logprobs]}


# --- the two formats must describe the same sequence -----------------------------------


def test_both_formats_agree_on_a_plain_sequence():
    tokens = [token(-0.1, -1.0), token(-0.3, -2.0)]
    assert process_openai_chat_completion(chat(tokens), 1) == process_openai_responses_api(responses(tokens))


# --- absent or partial logprob data ----------------------------------------------------


def test_a_choice_without_logprobs_yields_an_empty_sequence():
    assert process_openai_chat_completion({"choices": [{"logprobs": None}]}, 1) == [{}]


def test_a_choice_with_empty_content_yields_an_empty_sequence():
    assert process_openai_chat_completion({"choices": [{"logprobs": {"content": []}}]}, 1) == [{}]


def test_a_response_without_choices_yields_nothing():
    assert process_openai_chat_completion({"choices": []}, 1) == []


def test_an_output_item_without_content_yields_an_empty_sequence():
    assert process_openai_responses_api({"output": [{"content": []}]}) == [{}]


def test_a_content_part_without_logprobs_yields_an_empty_sequence():
    assert process_openai_responses_api({"output": [{"content": [{"logprobs": []}]}]}) == [{}]


def test_a_null_rank_logprob_is_dropped_rather_than_coerced():
    # None would become nan through float(); the parser must skip it instead
    entry = {"top_logprobs": [{"logprob": -0.5}, {"logprob": None}]}
    assert _extract_logprobs_from_token(entry) == [-0.5]


def test_a_token_with_no_ranks_extracts_nothing():
    assert _extract_logprobs_from_token({"top_logprobs": []}) == []


# --- iteration clamping ----------------------------------------------------------------


def test_more_iterations_than_choices_is_clamped():
    # `iterations` is caller-supplied; asking for more must not IndexError
    assert len(process_openai_chat_completion(chat([token(-0.1)], n_choices=2), iterations=5)) == 2


def test_fewer_iterations_than_choices_truncates():
    assert len(process_openai_chat_completion(chat([token(-0.1)], n_choices=3), iterations=1)) == 1


@pytest.mark.parametrize("iterations", [0, -1])
def test_non_positive_iterations_yield_nothing(iterations):
    assert process_openai_chat_completion(chat([token(-0.1)], n_choices=2), iterations) == []


# --- _parse_token_entry falls back to the sampled logprob ------------------------------


def test_a_token_entry_without_ranks_falls_back_to_its_own_logprob():
    assert _parse_token_entry({"logprob": -1.25}) == [-1.25]


def test_a_token_entry_with_neither_ranks_nor_logprob_is_empty():
    assert _parse_token_entry({}) == []


def test_ranks_take_precedence_over_the_sampled_logprob():
    assert _parse_token_entry({"logprob": -9.0, "top_logprobs": [{"logprob": -0.5}]}) == [-0.5]


def test_token_entry_ranks_come_back_descending():
    assert _parse_token_entry(token(-3.0, -0.5, -1.0)) == [-0.5, -1.0, -3.0]


# --- sampled-token logprobs ------------------------------------------------------------


def test_sampled_logprobs_skip_null_entries():
    tokens = [{"logprob": -0.1}, {"logprob": None}, {"logprob": -0.3}]
    (sampled,) = sampled_tokens_logprobs_responses_api(responses(tokens))
    np.testing.assert_allclose(sampled, [-0.1, -0.3])


def test_sampled_logprobs_for_a_choice_without_logprobs_are_empty():
    (sampled,) = sampled_tokens_logprobs_chat_completion_api({"choices": [{"logprobs": None}]})
    assert sampled.shape == (0,)


def test_sampled_logprobs_yield_one_array_per_choice():
    payload = chat([{"logprob": -0.1}], n_choices=3)
    assert len(sampled_tokens_logprobs_chat_completion_api(payload)) == 3


def test_sampled_logprobs_concatenate_multiple_content_parts():
    # a Responses output item may split its tokens over several content parts
    payload = {"output": [{"content": [{"logprobs": [{"logprob": -0.1}]}, {"logprobs": [{"logprob": -0.2}]}]}]}
    (sampled,) = sampled_tokens_logprobs_responses_api(payload)
    np.testing.assert_allclose(sampled, [-0.1, -0.2])


# --- _get_val --------------------------------------------------------------------------


def test_get_val_reads_an_attribute():
    class Holder:
        field = 7

    assert _get_val(Holder(), "field") == 7


def test_get_val_reads_a_mapping_key():
    assert _get_val({"field": 7}, "field") == 7


def test_get_val_returns_the_default_for_a_scalar():
    # neither attribute nor mapping: the default keeps the parsers total
    assert _get_val(42, "field", "fallback") == "fallback"


def test_get_val_defaults_to_none():
    assert _get_val(42, "field") is None


# --- is_openai_responses_api -----------------------------------------------------------


def test_a_payload_tagged_as_a_response_is_detected():
    assert is_openai_responses_api({"object": "response"})


def test_an_object_attribute_tagged_as_a_response_is_detected():
    class Payload:
        object = "response"

    assert is_openai_responses_api(Payload())


def test_a_payload_carrying_output_is_detected():
    assert is_openai_responses_api({"output": []})


def test_a_chat_completion_is_not_a_responses_payload():
    assert not is_openai_responses_api({"choices": []})


def test_an_unrelated_value_is_not_a_responses_payload():
    assert not is_openai_responses_api("nope")
