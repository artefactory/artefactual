"""`Judgment`: the verdict of an LLM-as-a-judge reply, read however the model wrapped it."""

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from artefactual.preprocessing import BatchRequestOutput, Judgment, read_judgment


def completion(content):
    return {"choices": [{"message": {"content": content}}]}


def line(custom_id, body, status_code=200):
    return BatchRequestOutput.model_validate({
        "custom_id": custom_id,
        "response": {"status_code": status_code, "body": body},
        "error": None,
    })


def fenced(text):
    return f"```json\n{text}\n```"


@given(verdict=st.booleans(), explanation=st.text())
def test_a_judgment_survives_the_json_it_is_asked_to_write(verdict, explanation):
    judgment = Judgment(judgment=verdict, explanation=explanation)
    assert Judgment.from_text(judgment.model_dump_json()) == judgment


@given(verdict=st.booleans())
def test_a_fenced_judgment_is_read_like_a_bare_one(verdict):
    assert Judgment.from_text(fenced(json.dumps({"judgment": verdict}))) == Judgment(judgment=verdict)


@pytest.mark.parametrize("wrap", [str, fenced], ids=["bare", "fenced"])
@pytest.mark.parametrize(
    ("body", "explanation"),
    [
        pytest.param('{"judgment":true}', "", id="compact-and-no-explanation"),
        pytest.param('{"judgment": true, "explanation": null}', "", id="null-explanation"),
        pytest.param('{"judgment": true, "explanation": 3}', "", id="non-string-explanation"),
        pytest.param('{"judgment": true, "explanation": ["a"]}', "", id="list-explanation"),
        pytest.param('{"judgment": true, "explanation": "matches"}', "matches", id="explained"),
    ],
)
def test_the_verdict_never_depends_on_the_explanation(wrap, body, explanation):
    assert Judgment.from_text(wrap(body)) == Judgment(judgment=True, explanation=explanation)


@pytest.mark.parametrize("wrap", [str, fenced], ids=["bare", "fenced"])
def test_a_stringified_boolean_is_not_a_verdict(wrap):
    """`bool("false")` is True: reading it as one would call every wrong answer correct."""
    assert Judgment.from_text(wrap('{"judgment": "false"}')) is None


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        pytest.param('Sure!\n```\n{"judgment": false}\n```\nHope that helps.', False, id="fence-in-prose"),
        pytest.param("true", True, id="bare-word"),
        pytest.param("  False.\n", False, id="bare-word-punctuated"),
        pytest.param('The verdict is "judgment": true, I think.', True, id="key-in-prose"),
        pytest.param('"judgment": false', False, id="key-without-braces"),
    ],
)
def test_a_reply_that_ignored_the_format_is_still_read(reply, verdict):
    assert Judgment.from_text(reply) == Judgment(judgment=verdict)


@pytest.mark.parametrize(
    "reply",
    [
        "I cannot decide.",
        "",
        "The answer is true, because it matches.",
        '{"verdict": true}',
        "[true]",
        '"judgment"',
        "```json\nnot json\n```",
        "1",
        '"true"',
        "yes",
    ],
)
def test_an_unreadable_reply_is_none_and_never_a_guess(reply):
    assert Judgment.from_text(reply) is None


def test_the_constructor_refuses_a_stringified_boolean():
    with pytest.raises(ValidationError):
        Judgment(judgment="false")


def test_it_is_the_schema_the_prompt_asks_for():
    assert set(Judgment.model_json_schema()["properties"]) == {"judgment", "explanation"}


def test_a_completion_is_read_through_its_message():
    assert Judgment.from_completion(completion('{"judgment": false, "explanation": "no"}')) == Judgment(
        judgment=False, explanation="no"
    )


@pytest.mark.parametrize(
    "payload",
    [None, {"error": {"message": "boom"}}, {"choices": []}, {"choices": [{"index": 0}]}],
    ids=["failed-line", "error-object", "no-choices", "no-message"],
)
def test_a_completion_with_no_reply_is_none(payload):
    assert Judgment.from_completion(payload) is None


def test_a_batch_is_read_one_entry_per_row_in_row_order():
    rows = [
        line("a", completion('{"judgment": true, "explanation": "yes"}')),
        line("b", {"error": {"message": "rejected"}}, status_code=400),
        line("c", completion("I cannot decide.")),
        line("d", completion(fenced('{"judgment": false}'))),
    ]

    assert Judgment.from_batch(rows) == [
        Judgment(judgment=True, explanation="yes"),
        None,
        None,
        Judgment(judgment=False),
    ]


def test_a_batch_line_with_a_non_http_error_is_none():
    row = BatchRequestOutput.model_validate({"custom_id": "a", "response": None, "error": {"code": "boom"}})
    assert Judgment.from_batch([row]) == [None]


def test_an_empty_batch_has_no_verdicts():
    assert Judgment.from_batch([]) == []


@given(verdict=st.booleans(), explanation=st.text())
def test_read_judgment_is_the_verdict_of_the_judgment(verdict, explanation):
    payload = completion(Judgment(judgment=verdict, explanation=explanation).model_dump_json())
    assert read_judgment(payload) is verdict
