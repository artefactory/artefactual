import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from artefactual.preprocessing.parser import (
    LogProbParser,
    _sampled_logprobs,
    _top_logprobs,
    parse_sampled_token_logprobs,
    parse_top_logprobs,
)

# The OpenAI dispatch cases that lived here asserted which processor got called, not what
# was parsed; they are covered end-to-end in test_openai_parsing.py.


def test_parse_top_logprobs_unsupported():
    with pytest.raises(TypeError, match="Unsupported output format"):
        parse_top_logprobs("unsupported_format")


# The OpenAI sampled-logprob dispatch cases are covered end-to-end in test_openai_parsing.py.


def test_parse_sampled_token_logprobs_unsupported():
    with pytest.raises(TypeError, match="Unsupported output format"):
        parse_sampled_token_logprobs("unsupported_format")


@st.composite
def chat_completion(draw):
    k = draw(st.integers(min_value=1, max_value=20))  # fixed top_logprobs count per response

    def token_entry():
        lps = draw(
            arrays(
                dtype=np.float32,
                shape=k,
                elements=st.floats(min_value=-30.0, max_value=0.0, allow_nan=False, allow_infinity=False, width=32),
            )
        )
        lps = np.sort(lps)[::-1]  # API returns top_logprobs descending
        return {"top_logprobs": [{"token": "x", "logprob": float(v)} for v in lps]}

    def choice():
        n_tokens = draw(st.integers(min_value=1, max_value=8))
        return {"logprobs": {"content": [token_entry() for _ in range(n_tokens)]}}

    n_choices = draw(st.integers(min_value=1, max_value=3))
    return {"choices": [choice() for _ in range(n_choices)]}


@given(chat_completion())
def test_transform_shape_and_padding(payload):
    arr = LogProbParser().transform(payload)
    assert arr.shape[0] == len(payload["choices"]) and arr.dtype == np.float64
    real = arr[~np.isnan(arr)]
    assert np.isfinite(real).all() and (real <= 0).all()  # NaN <-> padding invariant


@given(chat_completion(), st.data())
def test_transform_rejects_out_of_domain(payload, data):
    c = data.draw(st.integers(0, len(payload["choices"]) - 1))
    content = payload["choices"][c]["logprobs"]["content"]
    assume(content)
    t = data.draw(st.integers(0, len(content) - 1))
    bad = data.draw(st.sampled_from([float("inf"), float("nan"), 1.0]))
    content[t]["top_logprobs"][0]["logprob"] = bad
    with pytest.raises(ValueError, match="Invalid logprob"):
        LogProbParser().transform(payload)


# Statelessness - transform without having called fit first should work
def test_stateless():
    parser = LogProbParser()
    parser.transform({"choices": []})


# Sklearn round-trip
def test_sklearn_roundtrip():
    from sklearn.base import clone

    p = LogProbParser()
    assert p.get_params() == {"k": None}
    clone(p)


# Edge cases
def test_empty_batch_transforms_to_an_empty_cube():
    assert LogProbParser().transform([]).shape == (0, 0, 0)


# --- format dispatch -------------------------------------------------------------------
# Both extractors are singledispatch tables keyed on the validated response model. The
# fallback exists so an unregistered model fails loudly instead of returning nothing.


def test_top_logprob_dispatch_rejects_an_unregistered_response():
    with pytest.raises(TypeError, match="No top-logprob extractor registered"):
        _top_logprobs(object())


def test_sampled_logprob_dispatch_rejects_an_unregistered_response():
    with pytest.raises(TypeError, match="No sampled-logprob extractor registered"):
        _sampled_logprobs(object())


@pytest.mark.parametrize("payload", [None, 42, "text", {"unrelated": 1}])
def test_unsupported_payloads_are_rejected_by_both_entry_points(payload):
    with pytest.raises(TypeError, match="Unsupported output format"):
        parse_top_logprobs(payload)
    with pytest.raises(TypeError, match="Unsupported output format"):
        parse_sampled_token_logprobs(payload)


def test_a_content_part_without_logprobs_is_skipped():
    # a Responses item may interleave parts that carry no logprobs at all
    payload = {"output": [{"content": [{"logprobs": []}, {"logprobs": [{"logprob": -0.2}]}]}]}

    (sampled,) = parse_sampled_token_logprobs(payload)

    np.testing.assert_allclose(sampled, [-0.2])


def test_a_nested_batch_is_flattened_in_order():
    def one(logprob):
        return {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"logprob": logprob}]}]}}]}

    parsed = parse_top_logprobs([one(-0.1), [one(-0.2), one(-0.3)]])

    assert [sequence[0] for sequence in parsed] == [[-0.1], [-0.2], [-0.3]]


def test_a_tuple_batch_is_accepted():
    payload = {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"logprob": -0.1}]}]}}]}
    assert len(parse_top_logprobs((payload, payload))) == 2


# --- LogProbParser as a pipeline step --------------------------------------------------


def test_fit_returns_self_and_ignores_its_arguments():
    parser = LogProbParser()
    assert parser.fit(["anything"], ["a target"]) is parser


def test_a_batch_of_empty_sequences_transforms_to_an_empty_cube():
    # every choice parses to {}, so there is no token axis and no rank axis to size
    assert LogProbParser().transform({"choices": [{"logprobs": None}]}).shape == (1, 0, 0)


def test_a_response_without_logprobs_names_the_cause():
    # what a provider returns when logprobs were not requested: the shape parses, but no
    # position carries ranks, and the reduction downstream would fail on a zero-size axis
    with pytest.raises(ValueError, match="carry any log-probabilities"):
        LogProbParser(k=15).transform({"choices": [{"logprobs": None}]})


def test_an_empty_parse_without_a_declared_k_is_still_an_empty_cube():
    # no rank count was declared, so there is no expectation for the empty parse to break
    assert LogProbParser().transform({"choices": [{"logprobs": None}]}).shape == (1, 0, 0)
