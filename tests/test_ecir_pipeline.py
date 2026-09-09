"""Tests for the ECIR reproduction pipeline.

The pipeline replaces the paper's generation and judging scripts with `vllm run-batch`,
so the prompts moved out of Python and into template files rendered by `jq`. That is only
a faithful reproduction if the rendered text is byte-identical to what the original
jinja2 template produced -- these tests hold it to that.

`vllm` itself is not exercised: it has no darwin wheels and is not a dependency. The
batch envelopes here follow the documented OpenAI Batch shape that `run-batch` reads and
writes: `response` is `{status_code, request_id, body}` and the completion is the body.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ECIR = REPO / "scripts" / "ecir"
TRAIN = REPO / "scripts" / "train_detector.py"

jinja2 = pytest.importorskip("jinja2")

QUESTIONS = [
    {
        "question": "Who sent Augustine to England?",
        "question_id": "q-1",
        "short_answer": "Pope Gregory",
        "answer_aliases": ["Gregory I", "the Pope"],
    },
    # backslash, ampersand and quotes: a regex-based renderer would corrupt these
    {"question": 'Capital? A\\B & C "quoted"', "question_id": "q-2", "short_answer": "Paris", "answer_aliases": []},
    {"question": "One alias?", "question_id": "q-3", "short_answer": "X", "answer_aliases": ["Y"]},
]


def batch_line(custom_id, content, ranks=(-0.1, -0.9, -2.0)):
    """One line of a Batch output file: `response` is the envelope, the completion its body."""
    token = {"token": "t", "logprob": ranks[0], "top_logprobs": [{"token": "t", "logprob": r} for r in ranks]}
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "logprobs": {"content": [token, token]},
    }
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {"status_code": 200, "request_id": f"batch_{custom_id}", "body": {"choices": [choice]}},
        "error": None,
    }


@pytest.fixture
def pack(tmp_path):
    (tmp_path / "questions.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")
    responses = [batch_line(q["question_id"], f"answer for {q['question_id']}") for q in QUESTIONS]
    (tmp_path / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in responses) + "\n", encoding="utf-8")
    return tmp_path


def run(script, *args):
    result = subprocess.run([str(script), *map(str, args)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout


def run_jsonl(script, *args):
    """Run `script` and parse its stdout as JSON Lines."""
    return [json.loads(line) for line in run(script, *args).splitlines()]


@pytest.fixture(autouse=True)
def _needs_jq():
    if shutil.which("jq") is None:
        pytest.skip("jq is not installed")


# --- the prompts must survive the move out of Python -----------------------------------


def test_the_judge_prompt_is_byte_identical_to_the_jinja_template(pack):
    """The paper's judge prompt is a jinja2 template; jq now renders it.

    Whitespace matters: jinja runs with trim_blocks off, so every {% %} tag leaves the
    newline it sits on, and the template's trailing newline is dropped. A prompt that
    differs by a blank line is a different experiment.
    """
    template = jinja2.Template((ECIR / "prompts" / "judge.jinja").read_text(encoding="utf-8"))
    rendered = run(ECIR / "build_judge_requests.sh", pack / "questions.json", pack / "responses.jsonl", "judge-model")
    rows = [json.loads(line) for line in rendered.splitlines()]

    assert len(rows) == len(QUESTIONS)
    by_id = {q["question_id"]: q for q in QUESTIONS}
    for row in rows:
        question = by_id[row["custom_id"]]
        expected = template.render(
            query=question["question"],
            expected_answer=question["short_answer"],
            answer_aliases=question["answer_aliases"],
            generated_answer=f"answer for {row['custom_id']}",
        )
        assert row["body"]["messages"][0]["content"] == expected


def test_the_generation_prompt_keeps_the_original_indentation(pack):
    # the paper built it from an indented f-string, so lines 2-4 carry 12 leading spaces
    rows = run_jsonl(ECIR / "build_generation_requests.sh", pack / "questions.json", "m")

    content = rows[0]["body"]["messages"][0]["content"]
    assert content.startswith("You are a useful assistant")
    assert "\n            Please keep your output AS SHORT AND CONCISE AS POSSIBLE.\n" in content
    assert QUESTIONS[0]["question"] in content


# --- request shape ---------------------------------------------------------------------


def test_generation_requests_ask_for_logprobs(pack):
    rows = run_jsonl(ECIR / "build_generation_requests.sh", pack / "questions.json", "m", 15)

    assert [r["custom_id"] for r in rows] == [q["question_id"] for q in QUESTIONS]
    for row in rows:
        assert row["url"] == "/v1/chat/completions"
        assert row["body"]["logprobs"] is True
        assert row["body"]["top_logprobs"] == 15


def test_the_judge_gets_room_to_answer_in_json(pack):
    # the verdict is {"judgment": ..., "explanation": ...}, not a single token
    rows = run_jsonl(ECIR / "build_judge_requests.sh", pack / "questions.json", pack / "responses.jsonl", "j")

    assert all(r["body"]["max_completion_tokens"] > 1 for r in rows)
    assert all(r["body"]["temperature"] == 0 for r in rows)


def rejected_line(custom_id):
    """The failure shape that looks like success: a 4xx whose body is an error object.

    The spec reserves top-level `error` for non-HTTP failures, so a rejected request
    leaves it null. Filtering on `error` alone reads this line's error object as the
    model's answer and sends it to the judge.
    """
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 400,
            "request_id": f"batch_{custom_id}",
            "body": {"error": {"message": "context_length_exceeded", "type": "invalid_request_error"}},
        },
        "error": None,
    }


def test_a_rejected_generation_is_dropped_even_though_error_is_null(pack):
    """The status decides, not `error` -- otherwise the error object becomes an answer."""
    path = pack / "with_rejection.jsonl"
    path.write_text(json.dumps(rejected_line("q-4")) + "\n" + (pack / "responses.jsonl").read_text(), encoding="utf-8")

    result = subprocess.run(
        [str(ECIR / "build_judge_requests.sh"), str(pack / "questions.json"), str(path), "j"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["custom_id"] for row in rows] == [q["question_id"] for q in QUESTIONS]
    assert "dropping 1/" in result.stderr


def test_an_envelope_with_no_status_code_stops_the_run_and_names_the_line(pack):
    """An absent status is not evidence of success, so the script refuses rather than guesses."""
    unstatused = {"id": "batch_req_x", "custom_id": "q-9", "response": {"body": {"choices": []}}, "error": None}
    path = pack / "no_status.jsonl"
    path.write_text(json.dumps(unstatused) + "\n" + (pack / "responses.jsonl").read_text(), encoding="utf-8")

    result = subprocess.run(
        [str(ECIR / "build_judge_requests.sh"), str(pack / "questions.json"), str(path), "j"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "q-9" in result.stderr


def test_failed_generations_are_dropped_not_propagated(pack):
    failed = {"id": "batch_req_x", "custom_id": "q-1", "response": None, "error": "out of memory"}
    path = pack / "with_failure.jsonl"
    path.write_text(json.dumps(failed) + "\n" + (pack / "responses.jsonl").read_text(), encoding="utf-8")

    rows = run_jsonl(ECIR / "build_judge_requests.sh", pack / "questions.json", path, "j")

    assert len(rows) == len(QUESTIONS)


# --- training joins on custom_id, never on position ------------------------------------


def judgments_file(path, verdicts):
    lines = [batch_line(cid, json.dumps({"judgment": v, "explanation": "-"})) for cid, v in verdicts.items()]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_training_pairs_by_id_regardless_of_file_order(pack, tmp_path):
    """Reversing one file must not change the fitted model.

    Positional pairing would silently train on mismatched (response, verdict) pairs here,
    which is exactly the failure the id join exists to prevent.
    """
    import sys

    sys.path.insert(0, str(TRAIN.parent))
    import train_detector as tc

    verdicts = {"q-1": True, "q-2": False, "q-3": True}
    judgments = judgments_file(tmp_path / "judgments.jsonl", verdicts)

    forward = tc.read_batch_output(pack / "responses.jsonl")
    reversed_rows = "\n".join(reversed(judgments.read_text().strip().splitlines())) + "\n"
    (tmp_path / "reversed.jsonl").write_text(reversed_rows, encoding="utf-8")

    _, y_forward = tc.join_on_custom_id(forward, tc.read_batch_output(judgments))
    _, y_reversed = tc.join_on_custom_id(forward, tc.read_batch_output(tmp_path / "reversed.jsonl"))

    assert y_forward.tolist() == y_reversed.tolist() == [0, 1, 0]  # judgment True -> not a hallucination


def test_training_refuses_a_batch_with_no_shared_ids(pack, tmp_path):
    import sys

    sys.path.insert(0, str(TRAIN.parent))
    import train_detector as tc

    other = judgments_file(tmp_path / "other.jsonl", {"z-9": True})

    with pytest.raises(ValueError, match="No custom_id is present in both files"):
        tc.join_on_custom_id(tc.read_batch_output(pack / "responses.jsonl"), tc.read_batch_output(other))


def test_training_refuses_a_file_that_repeats_a_custom_id(pack, tmp_path):
    """A repeat would replace the first line and leave the pair count looking right.

    The id is what pairs a generation with its verdict, so two lines claiming the same one
    make the join ambiguous -- and the ambiguity resolves silently, which is why it raises
    here rather than being logged.
    """
    import sys

    sys.path.insert(0, str(TRAIN.parent))
    import train_detector as tc

    responses = (pack / "responses.jsonl").read_text(encoding="utf-8")
    doubled = tmp_path / "doubled.jsonl"
    doubled.write_text(responses + responses.splitlines()[0] + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="appears more than once"):
        tc.read_batch_output(doubled)


def test_training_refuses_a_repeat_whose_first_line_failed(pack, tmp_path):
    """The repeat is ambiguous whether or not either line produced a row.

    Checking only the ids that made it into the index would let this pair through, and the
    join would then silently take the second line's answer for that question.
    """
    import sys

    sys.path.insert(0, str(TRAIN.parent))
    import train_detector as tc

    first = json.loads((pack / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0])
    failed = json.dumps({"custom_id": first["custom_id"], "response": None, "error": {"message": "boom"}})
    doubled = tmp_path / "doubled.jsonl"
    doubled.write_text(failed + "\n" + (pack / "responses.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="appears more than once"):
        tc.read_batch_output(doubled)
