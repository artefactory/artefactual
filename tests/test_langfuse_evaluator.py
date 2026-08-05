"""Tests for the Langfuse trace evaluator.

`langfuse` is an optional dependency and the evaluator only ever touches three things on
the client — `api.trace.get`, `create_score` and the trace's `output`. Those are supplied
here by small hand-written stubs that record what they were called with, so the test runs
without the extra installed and asserts real behaviour rather than a mock's recollection.
"""

import numpy as np
import pytest
from conftest import write_json

from artefactual.adapters.langfuse.evaluator import HallucinationEvaluator
from artefactual.scoring import epr

CALIBRATION = {"intercept": 0.0, "coefficients": {"mean_entropy": 1.0}}


def chat_payload(n_ranks=3):
    ranks = [-3.0 - 0.1 * index for index in range(n_ranks)]
    return {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"logprob": v} for v in ranks]}]}}]}


class Trace:
    def __init__(self, output):
        self.output = output


class TraceApi:
    def __init__(self, trace):
        self._trace = trace
        self.requested = []

    def get(self, trace_id):
        self.requested.append(trace_id)
        return self._trace


class StubLangfuse:
    """The slice of the Langfuse client surface the evaluator actually uses."""

    def __init__(self, trace):
        self.api = type("Api", (), {"trace": TraceApi(trace)})()
        self.scores = []

    def create_score(self, **kwargs):
        self.scores.append(kwargs)


@pytest.fixture
def detector(tmp_path):
    return epr(str(write_json(tmp_path, "cal.json", CALIBRATION)))


@pytest.fixture
def client():
    return StubLangfuse(Trace(chat_payload()))


def test_scoring_returns_a_probability(client, detector):
    score = HallucinationEvaluator("epr", client, detector).score_trace("trace-1")

    assert 0.0 <= score <= 1.0


def test_the_trace_is_fetched_by_id(client, detector):
    HallucinationEvaluator("epr", client, detector).score_trace("trace-1")

    assert client.api.trace.requested == ["trace-1"]


def test_the_score_is_written_back_to_langfuse(client, detector):
    HallucinationEvaluator("epr", client, detector).score_trace("trace-1")

    (written,) = client.scores
    assert written["trace_id"] == "trace-1"
    assert written["name"] == "epr"


def test_the_written_value_matches_the_returned_score(client, detector):
    score = HallucinationEvaluator("epr", client, detector).score_trace("trace-1")

    assert client.scores[0]["value"] == pytest.approx(score)


def test_the_score_id_is_an_idempotency_key(client, detector):
    """Re-scoring an unchanged trace must reuse the same score id.

    The id is derived from the trace id and the value, so a repeat run overwrites rather
    than appending a duplicate score.
    """
    evaluator = HallucinationEvaluator("epr", client, detector)
    evaluator.score_trace("trace-1")
    evaluator.score_trace("trace-1")

    first, second = client.scores
    assert first["score_id"] == second["score_id"]


def test_the_evaluator_name_is_used_as_the_score_name(client, detector):
    HallucinationEvaluator("custom-name", client, detector).score_trace("trace-1")

    assert client.scores[0]["name"] == "custom-name"


def test_the_detector_reads_the_trace_output(detector):
    # a confident trace and an uncertain one must not receive the same score
    confident = StubLangfuse(Trace({"choices": [{"logprobs": {"content": [{"top_logprobs": [{"logprob": 0.0}]}]}}]}))
    uncertain = StubLangfuse(Trace(chat_payload()))

    confident_score = HallucinationEvaluator("epr", confident, detector).score_trace("a")
    uncertain_score = HallucinationEvaluator("epr", uncertain, detector).score_trace("b")

    assert confident_score != uncertain_score


def test_a_trace_without_logprobs_is_rejected(detector):
    # an output that carries no logprobs cannot be scored; it must not be silently zeroed
    client = StubLangfuse(Trace({"unrelated": "payload"}))

    with pytest.raises(TypeError, match="Unsupported output format"):
        HallucinationEvaluator("epr", client, detector).score_trace("trace-1")


def test_scoring_a_multi_sequence_trace_uses_the_first_sequence(detector):
    payload = chat_payload()
    payload["choices"].append(payload["choices"][0])
    client = StubLangfuse(Trace(payload))

    score = HallucinationEvaluator("epr", client, detector).score_trace("trace-1")

    expected = detector.predict_proba(payload)[0, 1]
    assert score == pytest.approx(float(expected))


def test_the_score_is_a_plain_float(client, detector):
    # langfuse serialises the value to JSON, which numpy scalars do not survive
    score = HallucinationEvaluator("epr", client, detector).score_trace("trace-1")

    assert type(score) is float
    assert not isinstance(score, np.floating)
