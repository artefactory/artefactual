"""Edge cases in the OpenAI extraction helpers.

`test_openai_parsing.py` covers the happy path over generated payloads. This file covers
the degenerate shapes real providers emit — absent `logprobs`, a token whose `top_logprobs`
came back empty, a `logprob` of `None` — and the places where the two wire formats are
supposed to mean the same thing.

The helpers take validated models, because `parse_top_logprobs` validates before it
dispatches to them. The builders below therefore go through the models too, so these tests
exercise the same objects production does rather than raw mappings the extractors never
see.
"""

import numpy as np
import pytest

from artefactual.preprocessing.openai_parser import (
    _ranks,
    _sampled_only,
    process_openai_chat_completion,
    process_openai_responses_api,
    sampled_tokens_logprobs_chat_completion_api,
    sampled_tokens_logprobs_responses_api,
)
from artefactual.preprocessing.response_models import (
    ChatCompletion,
    ResponsesPayload,
    TokenLogprobs,
)


def chat(tokens, n_choices=1):
    return ChatCompletion.model_validate({"choices": [{"logprobs": {"content": tokens}} for _ in range(n_choices)]})


def responses(tokens):
    return ResponsesPayload.model_validate({"output": [{"content": [{"logprobs": tokens}]}]})


def token(*logprobs):
    return {"top_logprobs": [{"logprob": value} for value in logprobs]}


def entry(payload):
    return TokenLogprobs.model_validate(payload)


# --- the two formats must describe the same sequence -----------------------------------


def test_both_formats_agree_on_a_plain_sequence():
    tokens = [token(-0.1, -1.0), token(-0.3, -2.0)]
    assert process_openai_chat_completion(chat(tokens)) == process_openai_responses_api(responses(tokens))


def test_both_formats_agree_when_a_token_has_no_top_logprobs():
    """A token with an empty `top_logprobs` must not shift the positions after it.

    Chat completion keys by `enumerate` position and skips the empty token, leaving a hole
    at index 1. The Responses parser keeps its own counter that only advances on non-empty
    tokens, so it compacts instead. Same content, two different token indexings — and
    `predict_token_proba` uses those indices to line scores up with the generated text.
    """
    tokens = [token(-0.1), {"top_logprobs": []}, token(-0.3)]

    from_chat = process_openai_chat_completion(chat(tokens))[0]
    from_responses = process_openai_responses_api(responses(tokens))[0]

    assert sorted(from_chat) == sorted(from_responses)


def test_a_dropped_token_does_not_renumber_later_tokens():
    # position 2 in the response must still be position 2 after parsing
    tokens = [token(-0.1), {"top_logprobs": []}, token(-0.3)]

    assert max(process_openai_responses_api(responses(tokens))[0]) == 2


# --- absent or partial logprob data ----------------------------------------------------


def test_a_choice_without_logprobs_yields_an_empty_sequence():
    payload = ChatCompletion.model_validate({"choices": [{"logprobs": None}]})
    assert process_openai_chat_completion(payload) == [{}]


def test_a_choice_with_empty_content_yields_an_empty_sequence():
    assert process_openai_chat_completion(chat([])) == [{}]


def test_a_response_without_choices_yields_nothing():
    payload = ChatCompletion.model_validate({"choices": []})
    assert process_openai_chat_completion(payload) == []


def test_an_output_item_without_content_yields_an_empty_sequence():
    payload = ResponsesPayload.model_validate({"output": [{"content": []}]})
    assert process_openai_responses_api(payload) == [{}]


def test_a_content_part_without_logprobs_yields_an_empty_sequence():
    assert process_openai_responses_api(responses([])) == [{}]


def test_a_null_rank_logprob_is_dropped_rather_than_coerced():
    # None would become nan through float(); the parser must skip it instead
    assert _ranks(entry({"top_logprobs": [{"logprob": -0.5}, {"logprob": None}]})) == [-0.5]


def test_a_token_with_no_ranks_extracts_nothing():
    assert _ranks(entry({"top_logprobs": []})) == []


def test_one_choice_per_sampled_sequence():
    assert len(process_openai_chat_completion(chat([token(-0.1)], n_choices=3))) == 3


# --- the Responses format falls back to the sampled logprob ----------------------------


def test_a_token_entry_without_ranks_falls_back_to_its_own_logprob():
    (sequence,) = process_openai_responses_api(responses([{"logprob": -1.25}]))
    assert sequence == {0: [-1.25]}


def test_a_token_entry_with_neither_ranks_nor_logprob_is_empty():
    assert _sampled_only(entry({})) == []


def test_ranks_take_precedence_over_the_sampled_logprob():
    payload = responses([{"logprob": -9.0, "top_logprobs": [{"logprob": -0.5}]}])
    (sequence,) = process_openai_responses_api(payload)
    assert sequence == {0: [-0.5]}


def test_token_entry_ranks_come_back_descending():
    assert _ranks(entry(token(-3.0, -0.5, -1.0))) == [-0.5, -1.0, -3.0]


# --- sampled-token logprobs ------------------------------------------------------------


def test_sampled_logprobs_skip_null_entries():
    tokens = [{"logprob": -0.1}, {"logprob": None}, {"logprob": -0.3}]
    (sampled,) = sampled_tokens_logprobs_responses_api(responses(tokens))
    np.testing.assert_allclose(sampled, [-0.1, -0.3])


def test_sampled_logprobs_for_a_choice_without_logprobs_are_empty():
    payload = ChatCompletion.model_validate({"choices": [{"logprobs": None}]})
    (sampled,) = sampled_tokens_logprobs_chat_completion_api(payload)
    assert sampled.shape == (0,)


def test_sampled_logprobs_yield_one_array_per_choice():
    payload = chat([{"logprob": -0.1}], n_choices=3)
    assert len(sampled_tokens_logprobs_chat_completion_api(payload)) == 3


def test_sampled_logprobs_concatenate_multiple_content_parts():
    # a Responses output item may split its tokens over several content parts
    payload = ResponsesPayload.model_validate({
        "output": [{"content": [{"logprobs": [{"logprob": -0.1}]}, {"logprobs": [{"logprob": -0.2}]}]}]
    })
    (sampled,) = sampled_tokens_logprobs_responses_api(payload)
    np.testing.assert_allclose(sampled, [-0.1, -0.2])


# --- the extractors are reached only through validated models --------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"logprob": -0.1}]}]}}]},
        {"object": "response", "output": [{"content": [{"logprobs": [{"top_logprobs": [{"logprob": -0.1}]}]}]}]},
    ],
)
def test_a_raw_mapping_goes_through_parse_top_logprobs(payload):
    # the public entry point validates, so callers still pass plain dicts; the extractors
    # themselves never see one
    from artefactual.preprocessing import parse_top_logprobs

    assert parse_top_logprobs(payload) == [{0: [-0.1]}]
