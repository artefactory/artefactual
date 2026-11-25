"""Edge cases in the logprob extraction the response models carry.

`test_openai_parsing.py` covers the happy path over generated payloads. This file covers
the degenerate shapes real providers emit — absent `logprobs`, a token whose `top_logprobs`
came back empty, a `logprob` of `None` — and the places where the two wire formats are
supposed to mean the same thing.

The extraction is a method of the validated model, so the builders below go through the
models, as production does.
"""

import numpy as np
import pytest

from artefactual.preprocessing.response_models import (
    BatchRequestOutput,
    ChatCompletion,
    LogprobsMixin,
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
    assert chat(tokens).top_logprobs() == responses(tokens).top_logprobs()


def test_both_formats_agree_when_a_token_has_no_top_logprobs():
    """A token with an empty `top_logprobs` must not shift the positions after it.

    Chat completion keys by `enumerate` position and skips the empty token, leaving a hole
    at index 1. The Responses parser keeps its own counter that only advances on non-empty
    tokens, so it compacts instead. Same content, two different token indexings — and
    `predict_token_proba` uses those indices to line scores up with the generated text.
    """
    tokens = [token(-0.1), {"top_logprobs": []}, token(-0.3)]

    from_chat = chat(tokens).top_logprobs()[0]
    from_responses = responses(tokens).top_logprobs()[0]

    assert sorted(from_chat) == sorted(from_responses)


def test_a_dropped_token_does_not_renumber_later_tokens():
    # position 2 in the response must still be position 2 after parsing
    tokens = [token(-0.1), {"top_logprobs": []}, token(-0.3)]

    assert max(responses(tokens).top_logprobs()[0]) == 2


# --- absent or partial logprob data ----------------------------------------------------


def test_a_choice_without_logprobs_yields_an_empty_sequence():
    payload = ChatCompletion.model_validate({"choices": [{"logprobs": None}]})
    assert payload.top_logprobs() == [{}]


def test_a_choice_with_empty_content_yields_an_empty_sequence():
    assert chat([]).top_logprobs() == [{}]


def test_a_choice_with_null_content_yields_an_empty_sequence():
    payload = ChatCompletion.model_validate({"choices": [{"logprobs": {"content": None}}]})
    assert payload.top_logprobs() == [{}]
    assert payload.sampled_logprobs()[0].tolist() == []


def test_a_response_without_choices_yields_nothing():
    payload = ChatCompletion.model_validate({"choices": []})
    assert payload.top_logprobs() == []


def test_an_output_item_without_content_yields_an_empty_sequence():
    payload = ResponsesPayload.model_validate({"output": [{"content": []}]})
    assert payload.top_logprobs() == [{}]


def test_a_content_part_without_logprobs_yields_an_empty_sequence():
    assert responses([]).top_logprobs() == [{}]


def test_a_null_rank_logprob_is_dropped_rather_than_coerced():
    # None would become nan through float(); the parser must skip it instead
    assert entry({"top_logprobs": [{"logprob": -0.5}, {"logprob": None}]}).candidates() == [-0.5]


def test_a_token_with_no_ranks_extracts_nothing():
    assert entry({"top_logprobs": []}).candidates() == []


def test_one_choice_per_sampled_sequence():
    assert len(chat([token(-0.1)], n_choices=3).top_logprobs()) == 3


# --- the Responses format falls back to the sampled logprob ----------------------------


def test_a_token_entry_without_ranks_falls_back_to_its_own_logprob():
    (sequence,) = responses([{"logprob": -1.25}]).top_logprobs()
    assert sequence == {0: [-1.25]}


def test_responses_sequences_are_the_validated_tokens_not_rebuilt_ones():
    payload = responses([{"logprob": -0.5, "top_logprobs": []}, token(-0.1, -1.0)])
    (sequence,) = payload.sequences()
    assert all(a is b for a, b in zip(sequence, payload.output[0].content[0].logprobs, strict=True))
    assert sequence[0].top_logprobs == []


def test_a_token_entry_with_neither_ranks_nor_logprob_is_empty():
    assert responses([{}]).top_logprobs() == [{}]


def test_ranks_take_precedence_over_the_sampled_logprob():
    payload = responses([{"logprob": -9.0, "top_logprobs": [{"logprob": -0.5}]}])
    (sequence,) = payload.top_logprobs()
    assert sequence == {0: [-0.5]}


def test_token_entry_ranks_come_back_descending():
    assert entry(token(-3.0, -0.5, -1.0)).candidates() == [-0.5, -1.0, -3.0]


# --- sampled-token logprobs ------------------------------------------------------------


def test_sampled_logprobs_skip_null_entries():
    tokens = [{"logprob": -0.1}, {"logprob": None}, {"logprob": -0.3}]
    (sampled,) = responses(tokens).sampled_logprobs()
    np.testing.assert_allclose(sampled, [-0.1, -0.3])


def test_sampled_logprobs_for_a_choice_without_logprobs_are_empty():
    payload = ChatCompletion.model_validate({"choices": [{"logprobs": None}]})
    (sampled,) = payload.sampled_logprobs()
    assert sampled.shape == (0,)


def test_sampled_logprobs_yield_one_array_per_choice():
    payload = chat([{"logprob": -0.1}], n_choices=3)
    assert len(payload.sampled_logprobs()) == 3


def test_sampled_logprobs_concatenate_multiple_content_parts():
    # a Responses output item may split its tokens over several content parts
    payload = ResponsesPayload.model_validate({
        "output": [{"content": [{"logprobs": [{"logprob": -0.1}]}, {"logprobs": [{"logprob": -0.2}]}]}]
    })
    (sampled,) = payload.sampled_logprobs()
    np.testing.assert_allclose(sampled, [-0.1, -0.2])


# --- parse: any input, one validated model ---------------------------------------------

CHAT_RAW = {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"logprob": -0.1}]}]}}]}
RESPONSES_RAW = {"object": "response", "output": [{"content": [{"logprobs": [{"top_logprobs": [{"logprob": -0.1}]}]}]}]}


def line(body, status_code=200, error=None):
    return {
        "custom_id": "q-7",
        "response": {"status_code": status_code, "body": body},
        "error": error,
    }


@pytest.mark.parametrize(
    ("raw", "model"),
    [
        pytest.param(CHAT_RAW, ChatCompletion, id="chat"),
        pytest.param(RESPONSES_RAW, ResponsesPayload, id="responses"),
        pytest.param(line(CHAT_RAW), ChatCompletion, id="chat-in-a-batch-line"),
        pytest.param(line(RESPONSES_RAW), ResponsesPayload, id="responses-in-a-batch-line"),
        pytest.param(BatchRequestOutput.model_validate(line(RESPONSES_RAW)), ResponsesPayload, id="line-object"),
        pytest.param(ChatCompletion.model_validate(CHAT_RAW), ChatCompletion, id="model"),
    ],
)
def test_parse_returns_the_format_the_input_carries(raw, model):
    parsed = LogprobsMixin.parse(raw)
    assert type(parsed) is model
    assert parsed.top_logprobs() == [{0: [-0.1]}]


def test_parse_of_a_model_is_the_model_itself():
    payload = ChatCompletion.model_validate(CHAT_RAW)
    assert LogprobsMixin.parse(payload) is payload


@pytest.mark.parametrize(
    ("model", "raw"),
    [
        pytest.param(ChatCompletion, CHAT_RAW, id="chat"),
        pytest.param(ChatCompletion, line(CHAT_RAW), id="chat-in-a-batch-line"),
        pytest.param(ResponsesPayload, RESPONSES_RAW, id="responses"),
        pytest.param(ResponsesPayload, line(RESPONSES_RAW), id="responses-in-a-batch-line"),
    ],
)
def test_a_format_parses_its_own_payload(model, raw):
    assert type(model.parse(raw)) is model


@pytest.mark.parametrize(
    ("model", "raw", "message"),
    [
        pytest.param(ChatCompletion, RESPONSES_RAW, "Unsupported output format", id="chat-given-responses"),
        pytest.param(ResponsesPayload, CHAT_RAW, "Unsupported output format", id="responses-given-chat"),
        pytest.param(ChatCompletion, line(RESPONSES_RAW), "not a ChatCompletion", id="chat-given-a-responses-line"),
        pytest.param(ResponsesPayload, line(CHAT_RAW), "not a ResponsesPayload", id="responses-given-a-chat-line"),
    ],
)
def test_a_format_refuses_the_other_format(model, raw, message):
    with pytest.raises(TypeError, match=message):
        model.parse(raw)


def test_the_formats_are_the_classes_implementing_the_mixin():
    assert LogprobsMixin.formats == [ChatCompletion, ResponsesPayload]


@pytest.mark.parametrize("raw", [None, 42, "text", {"unrelated": 1}, {"output": "nope"}, []])
def test_parse_names_an_unsupported_input(raw):
    with pytest.raises(TypeError, match="Unsupported output format"):
        LogprobsMixin.parse(raw)


def test_parse_keeps_a_recognised_payload_with_bad_data_a_validation_error():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LogprobsMixin.parse({"choices": [{"logprobs": {"content": [{"top_logprobs": [{"logprob": "abc"}]}]}}]})


# --- a batch line's payload ------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "why"),
    [
        pytest.param(line({"error": {"message": "boom"}}, status_code=429), "HTTP 429", id="rejected"),
        pytest.param({**line(CHAT_RAW), "error": {"code": "expired"}}, "expired", id="non-http-error"),
        pytest.param({"custom_id": "q-7", "response": None, "error": None}, "no response envelope", id="no-envelope"),
        pytest.param(line(None), "an empty response body", id="empty-body"),
    ],
)
def test_a_failed_line_has_no_payload_and_names_itself(record, why):
    with pytest.raises(ValueError, match=r"'q-7'.*" + why):
        BatchRequestOutput.model_validate(record).payload()


def test_a_body_that_is_no_completion_is_a_type_error_naming_the_line():
    with pytest.raises(TypeError, match="'q-7' carries a body that is neither"):
        BatchRequestOutput.model_validate(line({"zzz": 1})).payload()


def test_a_body_that_is_itself_a_batch_line_is_not_unwrapped_again():
    with pytest.raises(TypeError, match="neither a chat completion nor a Responses payload"):
        BatchRequestOutput.model_validate(line(line(CHAT_RAW))).payload()


def test_a_format_without_sequences_fails_loudly():
    with pytest.raises(NotImplementedError):
        LogprobsMixin().top_logprobs()
