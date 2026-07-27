from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from artefactual.preprocessing.parser import LogProbParser, parse_sampled_token_logprobs, parse_top_logprobs


class MockVLLMOutput:
    def __init__(self, outputs):
        self.outputs = outputs


# The OpenAI dispatch cases that lived here asserted which processor got called, not what
# was parsed; they are covered end-to-end in test_openai_parsing.py.


def test_parse_top_logprobs_no_longer_accepts_vllm_outputs():
    # standard completion responses only; vLLM outputs must be converted first
    with pytest.raises(TypeError, match="Unsupported output format"):
        parse_top_logprobs([MockVLLMOutput(outputs=[])])


def test_parse_top_logprobs_unsupported():
    with pytest.raises(TypeError, match="Unsupported output format"):
        parse_top_logprobs("unsupported_format")


@patch("artefactual.preprocessing.parser.vllm_sampled_tokens_logprobs")
def test_parse_sampled_token_logprobs_vllm(mock_process):
    mock_process.return_value = [np.array([-0.1, -0.2])]
    outputs = [MockVLLMOutput(outputs=[1, 2])]

    result = parse_sampled_token_logprobs(outputs)

    mock_process.assert_called_once_with(outputs, 2)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], np.array([-0.1, -0.2]))


def test_parse_sampled_token_logprobs_vllm_empty():
    outputs = [MockVLLMOutput(outputs=[])]
    result = parse_sampled_token_logprobs(outputs)
    assert result == []


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
    assert arr.shape[0] == len(payload["choices"]) and arr.dtype == np.float32
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
    assert p.get_params() == {}
    clone(p)


# Edge cases
def test_empty_batch_transforms_to_an_empty_cube():
    assert LogProbParser().transform([]).shape == (0, 0, 0)
