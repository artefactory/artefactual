"""Tests for the OpenAI Batch output envelope.

The Batch API returns JSONL -- one `BatchRequestOutput` per line -- and any
OpenAI-compatible server writes the same shape. These hold the model to it, and in
particular to the two ways the spec reports a failed request.
"""

import json
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from artefactual.preprocessing.parser import _RESPONSE_ADAPTER, LogProbParser
from artefactual.preprocessing.response_models import BatchRequestOutput

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Sunset Boulevard"},
            "logprobs": {
                "content": [{"token": "S", "logprob": -0.1, "top_logprobs": [{"token": "S", "logprob": -0.1}]}]
            },
        }
    ],
}


def line(**overrides):
    record = {"id": "batch_req_1", "custom_id": "q-1", "response": None, "error": None}
    record.update(overrides)
    return record


def test_the_spec_envelope_yields_its_completion():
    record = BatchRequestOutput.model_validate(
        line(response={"status_code": 200, "request_id": "batch-1", "body": COMPLETION})
    )

    assert record.custom_id == "q-1"
    assert record.response.status_code == 200
    assert record.completion == COMPLETION


def test_a_failed_request_carries_no_completion():
    record = BatchRequestOutput.model_validate(line(response=None, error={"message": "boom"}))

    assert record.completion is None
    assert record.custom_id == "q-1"  # still joinable, so the failure can be reported against it


def test_an_error_envelope_without_a_body_carries_no_completion():
    """The non-HTTP failure shape: `error` is set and the envelope carries no body."""
    record = BatchRequestOutput.model_validate(
        line(response={"status_code": 400, "request_id": "batch-1"}, error={"message": "bad request"})
    )

    assert record.completion is None


def test_a_rejected_request_carries_no_completion_even_though_it_has_a_body():
    """The failure shape that looks like success.

    The Batch API documents top-level `error` as non-HTTP failures only. A request the API
    rejects comes back with `error: null`, a 4xx status and an *error object* where the
    completion would be -- so a body is not evidence of a completion, and reading one as a
    completion puts a row of nothing into the training data.
    """
    record = BatchRequestOutput.model_validate(
        line(
            response={
                "status_code": 400,
                "request_id": "batch-1",
                "body": {"error": {"message": "context_length_exceeded", "type": "invalid_request_error"}},
            },
            error=None,
        )
    )

    assert record.completion is None
    assert record.failure == "HTTP 400"


def test_a_successful_envelope_with_no_body_is_reported_as_empty_rather_than_as_a_failure():
    """A 200 that carries nothing did not fail; the message should say which of the two."""
    record = BatchRequestOutput.model_validate(line(response={"status_code": 200}))

    assert record.completion is None
    assert record.failure == "an empty response body"


def test_custom_id_is_required():
    """It is the only id that says which request this was; the rest are provider-assigned."""
    with pytest.raises(ValidationError):
        BatchRequestOutput.model_validate({"id": "batch_req_1", "response": None, "error": None})


def test_a_line_parses_straight_from_json():
    record = BatchRequestOutput.model_validate_json(json.dumps(line(response={"status_code": 200, "body": COMPLETION})))

    assert record.completion == COMPLETION


def test_unmodelled_fields_are_ignored():
    """Providers add fields; a reader should not have to enumerate them."""
    envelope = {"status_code": 200, "body": COMPLETION, "headers": {"x-request-id": "abc"}}
    record = BatchRequestOutput.model_validate(line(response=envelope, unexpected="ignored"))

    assert record.completion == COMPLETION


# --- the parser reads a batch line as readily as a plain completion -------------------


def wide(k=3):
    """One token carrying `k` ranks."""
    ranks = [{"token": f"t{i}", "logprob": -0.1 * (i + 1)} for i in range(k)]
    return {"token": "t", "logprob": -0.1, "top_logprobs": ranks}


def completion(k=3):
    choice = {"index": 0, "message": {"role": "assistant", "content": "a"}, "logprobs": {"content": [wide(k)]}}
    return {"choices": [choice]}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(completion(), "ChatCompletion", id="plain-completion"),
        pytest.param(line(response={"status_code": 200, "body": completion()}), "BatchRequestOutput", id="batch-line"),
        pytest.param({"output": [{"content": [{"logprobs": [wide()]}]}]}, "ResponsesPayload", id="responses-payload"),
    ],
)
def test_each_payload_validates_as_its_own_type(payload, expected):
    """The union must not coerce one format into another: a batch line has no top-level
    `choices` and a completion has no `custom_id`, which is what keeps them apart."""
    assert type(_RESPONSE_ADAPTER.validate_python(payload)).__name__ == expected


def test_a_batch_line_parses_without_being_unwrapped():
    """A batch output file can be fed to the parser as it is read."""
    envelope = {"status_code": 200, "request_id": "r", "body": completion()}
    record = BatchRequestOutput.model_validate(line(response=envelope))

    assert LogProbParser(k=3).transform([record]).shape == (1, 1, 3)


def test_batch_lines_and_completions_mix_in_one_batch():
    payloads = [completion(), line(response={"status_code": 200, "body": completion()}), completion()]

    assert LogProbParser(k=3).transform(payloads).shape == (3, 1, 3)


def test_a_failed_line_is_refused_rather_than_dropped():
    """Dropping it would shift every later response against its label, which no error
    would surface -- so the parser refuses and names the id."""
    failed = line(response=None, error={"message": "upstream timeout"})

    with pytest.raises(ValueError, match="q-1"):
        LogProbParser(k=3).transform([failed])


def test_a_rejected_line_is_refused_by_the_parser_too():
    """The 4xx-with-a-body shape reaches the parser looking like any other line."""
    rejected = line(response={"status_code": 429, "body": {"error": {"message": "rate limited"}}})

    with pytest.raises(ValueError, match=r"q-1.*HTTP 429"):
        LogProbParser(k=3).transform([rejected])


def test_a_body_that_is_not_a_completion_names_the_line_it_came_from():
    """Validating the body inside the union would raise about `choices` and no `custom_id`,
    which is unusable against a file of thousands of lines.

    The pydantic error stays as the cause, so the detail is one `__cause__` away.
    """
    with pytest.raises(TypeError, match="q-1") as raised:
        LogProbParser(k=3).transform([line(response={"status_code": 200, "body": {"unexpected": "shape"}})])

    assert isinstance(raised.value.__cause__, ValidationError)


def test_a_batch_line_carrying_a_responses_payload_parses():
    """Batch takes `/v1/responses` as an endpoint too, and this package models that payload,
    so the body is dispatched rather than assumed to be a chat completion."""
    payload = {"output": [{"content": [{"logprobs": [wide()]}]}]}

    assert LogProbParser(k=3).transform([line(response={"status_code": 200, "body": payload})]).shape == (1, 1, 3)


def test_the_sampled_logprob_path_reads_a_batch_line_too():
    """`parse_top_logprobs` and `parse_sampled_token_logprobs` unwrap the line separately,
    so both entry points need holding to it."""
    from artefactual.preprocessing.parser import parse_sampled_token_logprobs

    sampled = parse_sampled_token_logprobs(line(response={"status_code": 200, "body": completion()}))

    assert len(sampled) == 1


def test_a_mapping_that_only_carries_a_custom_id_is_not_a_batch_line():
    """Every line of a Batch output file says how the request went, in `response` or `error`.

    Without that, any unrecognised payload carrying a `custom_id` validates into this model
    and is reported as a batch line whose request failed -- an answer about the wrong thing,
    where "this is not a format I know" is the truth.
    """
    with pytest.raises(ValidationError):
        BatchRequestOutput.model_validate({"custom_id": "q-1"})


def test_an_empty_custom_id_is_refused():
    """It is the key every stage joins on; an empty one joins everything to everything."""
    with pytest.raises(ValidationError):
        BatchRequestOutput.model_validate(line(custom_id="", response={"body": COMPLETION}))


@pytest.mark.parametrize(
    "record",
    [
        pytest.param({"custom_id": "q-1", "error": None}, id="no-response-key"),
        pytest.param({"custom_id": "q-1", "response": None}, id="no-error-key"),
        pytest.param(SimpleNamespace(custom_id="q-1", method="POST", url="/v1/chat/completions", body={}), id="object"),
    ],
)
def test_a_payload_carrying_neither_response_nor_error_is_not_a_batch_line(record):
    """Both keys are required, so the rule holds for an object as well as a mapping.

    A `BatchRequestInput` object fed here by mistake carries a `custom_id` and neither of
    them. Without the requirement it validated into this model and was reported as a batch
    line whose request failed -- an answer about the wrong thing.
    """
    with pytest.raises(ValidationError):
        BatchRequestOutput.model_validate(record)


def test_an_envelope_with_no_status_code_is_refused():
    """An absent status is not evidence of success: the body could be an error object."""
    with pytest.raises(ValidationError):
        BatchRequestOutput.model_validate(line(response={"body": {"error": {"message": "boom"}}}))


# --- the invariant the two properties share --------------------------------------------


@given(
    status=st.integers(min_value=0, max_value=999),
    body=st.none() | st.dictionaries(st.text(max_size=8), st.integers(), max_size=3),
    error=st.none() | st.text(max_size=16) | st.dictionaries(st.text(max_size=8), st.text(max_size=8), max_size=2),
)
def test_a_line_carries_a_completion_exactly_when_no_failure_is_named(status, body, error):
    """`completion` and `failure` are two readings of one question, and must never disagree.

    Every caller branches on one or the other -- `read_batch_output` counts on `completion`
    being `None`, the parser's message quotes `failure` -- so a line that reported a
    completion and a reason for having none would put an error object into training data
    under a label that looks fine.
    """
    record = BatchRequestOutput.model_validate(line(response={"status_code": status, "body": body}, error=error))

    assert (record.completion is None) == (record.failure is not None)
    if record.completion is not None:
        assert error is None
        assert 200 <= status < 300
        assert body is not None


@given(error=st.none() | st.text(max_size=16))
def test_a_line_with_no_envelope_never_carries_a_completion(error):
    """`response: null` is how the spec reports a non-HTTP failure, with or without `error`."""
    record = BatchRequestOutput.model_validate(line(response=None, error=error))

    assert record.completion is None
    assert record.failure is not None
