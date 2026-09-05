"""Tests for the OpenAI Batch output envelope.

The Batch API returns JSONL -- one `BatchRequestOutput` per line -- and `vllm run-batch`
writes the same shape. Neither the OpenAI SDK nor this package modelled it before, so
every reader unwrapped it by hand; these hold the model to the shapes actually in the
wild, including the older vllm lines that put the completion straight in `response`.
"""

import json
from types import SimpleNamespace

import pytest
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
    record = {"id": "vllm-1", "custom_id": "q-1", "response": None, "error": None}
    record.update(overrides)
    return record


def test_the_spec_envelope_yields_its_completion():
    record = BatchRequestOutput.model_validate(
        line(response={"status_code": 200, "request_id": "batch-1", "body": COMPLETION})
    )

    assert record.custom_id == "q-1"
    assert record.response.status_code == 200
    assert record.completion == COMPLETION


def test_a_bare_completion_is_accepted_as_older_vllm_wrote_it():
    """Versions before the Batch spec put the ChatCompletion directly in `response`."""
    record = BatchRequestOutput.model_validate(line(response=COMPLETION))

    assert record.completion == COMPLETION


def test_a_failed_request_carries_no_completion():
    record = BatchRequestOutput.model_validate(line(response=None, error={"message": "boom"}))

    assert record.completion is None
    assert record.custom_id == "q-1"  # still joinable, so the failure can be reported against it


def test_an_error_envelope_without_a_body_carries_no_completion():
    """vllm fills status_code and omits the body when the request itself failed."""
    record = BatchRequestOutput.model_validate(
        line(response={"status_code": 400, "request_id": "batch-1"}, error={"message": "bad request"})
    )

    assert record.completion is None


def test_a_rejected_request_carries_no_completion_even_though_it_has_a_body():
    """The OpenAI failure shape, which is not vllm's and is the one that looks like success.

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


def test_an_empty_envelope_is_reported_as_empty_rather_than_as_a_failure():
    """`response: {}` did not fail; it carries nothing, and the message should say which."""
    record = BatchRequestOutput.model_validate(line(response={}))

    assert record.completion is None
    assert record.failure == "an empty response body"


def test_custom_id_is_required():
    """It is the only id that says which request this was; the rest are provider-assigned."""
    with pytest.raises(ValidationError):
        BatchRequestOutput.model_validate({"id": "vllm-1", "response": None, "error": None})


def test_a_line_parses_straight_from_json():
    record = BatchRequestOutput.model_validate_json(json.dumps(line(response={"body": COMPLETION})))

    assert record.completion == COMPLETION


def test_unmodelled_fields_are_ignored():
    """Providers add fields; a reader should not have to enumerate them."""
    record = BatchRequestOutput.model_validate(
        line(response={"body": COMPLETION, "headers": {"x-request-id": "abc"}}, unexpected="ignored")
    )

    assert record.completion == COMPLETION


def test_the_bare_completion_rule_does_not_catch_an_envelope_with_choices():
    """An envelope is recognised by `body`, which is checked before `choices`."""
    record = BatchRequestOutput.model_validate(line(response={"body": COMPLETION, "choices": "not the completion"}))

    assert record.completion == COMPLETION


# --- the parser reads a batch line as readily as a bare completion --------------------


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
        pytest.param(completion(), "ChatCompletion", id="bare-completion"),
        pytest.param(line(response={"status_code": 200, "body": completion()}), "BatchRequestOutput", id="batch-line"),
        pytest.param({"output": [{"content": [{"logprobs": [wide()]}]}]}, "ResponsesPayload", id="responses-payload"),
    ],
)
def test_each_payload_validates_as_its_own_type(payload, expected):
    """The union must not coerce one format into another: a batch line has no top-level
    `choices` and a completion has no `custom_id`, which is what keeps them apart."""
    assert type(_RESPONSE_ADAPTER.validate_python(payload)).__name__ == expected


@pytest.mark.parametrize(
    "response",
    [
        pytest.param({"status_code": 200, "request_id": "r", "body": completion()}, id="spec-envelope"),
        pytest.param(completion(), id="bare-completion-in-response"),
    ],
)
def test_a_batch_line_parses_without_being_unwrapped(response):
    """A run-batch output file can be fed to the parser as it is read."""
    assert LogProbParser(k=3).transform([BatchRequestOutput.model_validate(line(response=response))]).shape == (1, 1, 3)


def test_batch_lines_and_completions_mix_in_one_batch():
    payloads = [completion(), line(response={"body": completion()}), completion()]

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
    which is unusable against a file of thousands of lines."""
    with pytest.raises(TypeError, match="q-1"):
        LogProbParser(k=3).transform([line(response={"body": {"unexpected": "shape"}})])


def test_a_batch_line_carrying_a_responses_payload_parses():
    """Batch takes `/v1/responses` as an endpoint too, and this package models that payload,
    so the body is dispatched rather than assumed to be a chat completion."""
    payload = {"output": [{"content": [{"logprobs": [wide()]}]}]}

    assert LogProbParser(k=3).transform([line(response={"body": payload})]).shape == (1, 1, 3)


def object_completion(k=3):
    """The same completion as attributes rather than keys, as an SDK object arrives."""
    ranks = [SimpleNamespace(token=f"t{i}", logprob=-0.1 * (i + 1)) for i in range(k)]
    token = SimpleNamespace(token="t", logprob=-0.1, top_logprobs=ranks)
    return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[token]))])


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(
            SimpleNamespace(id="vllm-1", custom_id="q-1", response=object_completion(), error=None),
            id="object-line",
        ),
        pytest.param(line(response=object_completion()), id="dict-line-object-response"),
    ],
)
def test_a_bare_completion_is_recognised_when_it_is_an_object(record):
    """An in-process vllm `BatchRequestOutput` is an object, not a mapping. Sniffing for
    keys only would leave it with no body -- reported as a request that never failed."""
    assert LogProbParser(k=3).transform([record]).shape == (1, 1, 3)


def test_the_sampled_logprob_path_reads_a_batch_line_too():
    """`_top_logprobs` and `_sampled_logprobs` are separate dispatches; both are entry
    points, and only one of them was covered."""
    from artefactual.preprocessing.parser import parse_sampled_token_logprobs

    sampled = parse_sampled_token_logprobs(line(response={"body": completion()}))

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
