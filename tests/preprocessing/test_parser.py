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


class MockOpenAIChatCompletion:
    def __init__(self, choices):
        self.choices = choices


class MockOpenAIResponsesAPI:
    def __init__(self):
        self.object = "response"


@patch("artefactual.preprocessing.parser.process_vllm_top_logprobs")
def test_parse_top_logprobs_vllm(mock_process):
    mock_process.return_value = [{0: [-0.1]}]
    outputs = [MockVLLMOutput(outputs=[1, 2])]

    result = parse_top_logprobs(outputs)

    mock_process.assert_called_once_with(outputs, 2)
    assert result == [{0: [-0.1]}]


def test_parse_top_logprobs_vllm_empty():
    outputs = [MockVLLMOutput(outputs=[])]
    result = parse_top_logprobs(outputs)
    assert result == []


@patch("artefactual.preprocessing.parser.process_openai_chat_completion")
def test_parse_top_logprobs_openai_chat_completion_obj(mock_process):
    mock_process.return_value = [{0: [-0.2]}]
    outputs = MockOpenAIChatCompletion(choices=[1, 2, 3])

    result = parse_top_logprobs(outputs)

    mock_process.assert_called_once_with(outputs, iterations=3)
    assert result == [{0: [-0.2]}]


@patch("artefactual.preprocessing.parser.process_openai_chat_completion")
def test_parse_top_logprobs_openai_chat_completion_dict(mock_process):
    mock_process.return_value = [{0: [-0.3]}]
    outputs = {"choices": [1, 2]}

    result = parse_top_logprobs(outputs)

    mock_process.assert_called_once_with(outputs, iterations=2)
    assert result == [{0: [-0.3]}]


@patch("artefactual.preprocessing.parser.is_openai_responses_api")
@patch("artefactual.preprocessing.parser.process_openai_responses_api")
def test_parse_top_logprobs_openai_responses_api(mock_process, mock_is_responses):
    mock_is_responses.return_value = True
    mock_process.return_value = [{0: [-0.4]}]
    outputs = MockOpenAIResponsesAPI()

    result = parse_top_logprobs(outputs)

    mock_process.assert_called_once_with(outputs)
    assert result == [{0: [-0.4]}]


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


@patch("artefactual.preprocessing.parser.sampled_tokens_logprobs_chat_completion_api")
def test_parse_sampled_token_logprobs_openai_chat_completion(mock_process):
    mock_process.return_value = [np.array([-0.3])]
    outputs = MockOpenAIChatCompletion(choices=[1])

    result = parse_sampled_token_logprobs(outputs)

    mock_process.assert_called_once_with(outputs)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], np.array([-0.3]))


@patch("artefactual.preprocessing.parser.is_openai_responses_api")
@patch("artefactual.preprocessing.parser.sampled_tokens_logprobs_responses_api")
def test_parse_sampled_token_logprobs_openai_responses_api(mock_process, mock_is_responses):
    mock_is_responses.return_value = True
    mock_process.return_value = [np.array([-0.4])]
    outputs = MockOpenAIResponsesAPI()

    result = parse_sampled_token_logprobs(outputs)

    mock_process.assert_called_once_with(outputs)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], np.array([-0.4]))


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
def test_edge_cases():
    class MockCompletionOutput:
        logprobs = None

    assert LogProbParser().transform([MockVLLMOutput(outputs=[])]).shape == (0, 0, 0)  # réponse vide
    assert LogProbParser().transform([MockVLLMOutput(outputs=[MockCompletionOutput()])]).shape == (
        1,
        0,
        0,
    )  # réponse sans logprobs
