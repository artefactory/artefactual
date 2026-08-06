"""Tests for the Langfuse trace evaluator.

`langfuse` is an optional dependency and the evaluator only ever touches three things on
the client — `api.trace.get`, `create_score` and the trace's `output`. Those are supplied
here by small hand-written stubs that record what they were called with, so the test runs
without the extra installed and asserts real behaviour rather than a mock's recollection.

Trace payloads and estimators are drawn rather than fixed: the evaluator must behave the
same for any response the pipeline accepts, and a single hand-written payload only ever
proves it for that one.
"""

import numpy as np
import pytest
from conftest import chat_payloads_of_fixed_width, estimators, write_estimator
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from artefactual.adapters.langfuse.evaluator import HallucinationEvaluator
from artefactual.scoring import epr
from artefactual.scoring.base_detector import DEFAULT_K

# The detector defaults to DEFAULT_K, and the parser refuses anything narrower.
traces = chat_payloads_of_fixed_width(min_ranks=DEFAULT_K, max_ranks=20)
drawn = settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)


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


def build_detector(tmp_path, detector):
    return epr(str(write_estimator(tmp_path, "cal.skops", detector)))


@drawn
@given(payload=traces, detector=estimators(n_features=1))
def test_scoring_returns_a_probability(tmp_path, payload, detector):
    client = StubLangfuse(Trace(payload))

    score = HallucinationEvaluator("epr", client, build_detector(tmp_path, detector)).score_trace("trace-1")

    assert 0.0 <= score <= 1.0


@drawn
@given(payload=traces, detector=estimators(n_features=1))
def test_the_trace_is_fetched_by_id(tmp_path, payload, detector):
    client = StubLangfuse(Trace(payload))

    HallucinationEvaluator("epr", client, build_detector(tmp_path, detector)).score_trace("trace-1")

    assert client.api.trace.requested == ["trace-1"]


@drawn
@given(payload=traces, detector=estimators(n_features=1))
def test_the_score_is_written_back_to_langfuse(tmp_path, payload, detector):
    client = StubLangfuse(Trace(payload))

    HallucinationEvaluator("epr", client, build_detector(tmp_path, detector)).score_trace("trace-1")

    (written,) = client.scores
    assert written["trace_id"] == "trace-1"
    assert written["name"] == "epr"


@drawn
@given(payload=traces, detector=estimators(n_features=1))
def test_the_written_value_matches_the_returned_score(tmp_path, payload, detector):
    client = StubLangfuse(Trace(payload))

    score = HallucinationEvaluator("epr", client, build_detector(tmp_path, detector)).score_trace("trace-1")

    assert client.scores[0]["value"] == pytest.approx(score)


@drawn
@given(payload=traces, detector=estimators(n_features=1))
def test_the_score_id_is_an_idempotency_key(tmp_path, payload, detector):
    """Re-scoring an unchanged trace must reuse the same score id.

    The id is derived from the trace id and the value, so a repeat run overwrites rather
    than appending a duplicate score.
    """
    client = StubLangfuse(Trace(payload))
    evaluator = HallucinationEvaluator("epr", client, build_detector(tmp_path, detector))

    evaluator.score_trace("trace-1")
    evaluator.score_trace("trace-1")

    first, second = client.scores
    assert first["score_id"] == second["score_id"]


@drawn
@given(payload=traces, first_calibration=estimators(n_features=1), second_calibration=estimators(n_features=1))
def test_the_score_id_survives_a_change_of_value(tmp_path, payload, first_calibration, second_calibration):
    """A re-score that produces a *different* value must still overwrite.

    This is what scheduling the evaluator depends on: retraining the detector, or swapping
    it, moves the score. Keying the id on the value would mint a fresh id each time and
    accumulate rows instead of replacing the previous one.
    """
    client = StubLangfuse(Trace(payload))
    for index, detector in enumerate((first_calibration, second_calibration)):
        detector = epr(str(write_estimator(tmp_path, f"cal-{index}.skops", detector)))
        HallucinationEvaluator("epr", client, detector).score_trace("trace-1")

    first, second = client.scores
    assert first["score_id"] == second["score_id"]


@drawn
@given(payload=traces, detector=estimators(n_features=1))
def test_the_evaluator_name_is_used_as_the_score_name(tmp_path, payload, detector):
    client = StubLangfuse(Trace(payload))

    HallucinationEvaluator("custom-name", client, build_detector(tmp_path, detector)).score_trace("trace-1")

    assert client.scores[0]["name"] == "custom-name"


@drawn
@given(payload=traces, detector=estimators(n_features=1))
def test_the_score_is_the_detectors_own_verdict_on_the_trace_output(tmp_path, payload, detector):
    # the evaluator must score what the trace carried, not a re-derived or default input
    detector = build_detector(tmp_path, detector)
    client = StubLangfuse(Trace(payload))

    score = HallucinationEvaluator("epr", client, detector).score_trace("trace-1")

    assert score == pytest.approx(float(detector.predict_proba(payload)[0, 1]))


@drawn
@given(output=st.dictionaries(st.text(min_size=1), st.text(), min_size=1), detector=estimators(n_features=1))
def test_a_trace_without_logprobs_is_rejected(tmp_path, output, detector):
    # an output that carries no logprobs cannot be scored; it must not be silently zeroed
    client = StubLangfuse(Trace(output))

    with pytest.raises(TypeError, match="Unsupported output format"):
        HallucinationEvaluator("epr", client, build_detector(tmp_path, detector)).score_trace("trace-1")


@drawn
@given(payload=traces, detector=estimators(n_features=1))
def test_scoring_a_multi_sequence_trace_uses_the_first_sequence(tmp_path, payload, detector):
    detector = build_detector(tmp_path, detector)
    duplicated = {"choices": [*payload["choices"], payload["choices"][0]]}
    client = StubLangfuse(Trace(duplicated))

    score = HallucinationEvaluator("epr", client, detector).score_trace("trace-1")

    assert score == pytest.approx(float(detector.predict_proba(duplicated)[0, 1]))


@drawn
@given(payload=traces, detector=estimators(n_features=1))
def test_the_score_is_a_plain_float(tmp_path, payload, detector):
    # langfuse serialises the value to JSON, which numpy scalars do not survive
    client = StubLangfuse(Trace(payload))

    score = HallucinationEvaluator("epr", client, build_detector(tmp_path, detector)).score_trace("trace-1")

    assert type(score) is float
    assert not isinstance(score, np.floating)
