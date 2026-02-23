from unittest.mock import patch

import numpy as np
import pytest

from artefactual.preprocessing.parser import parse_sampled_token_logprobs, parse_top_logprobs


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
    mock_process.return_value = np.array([-0.1, -0.2])
    outputs = [MockVLLMOutput(outputs=[1, 2])]

    result = parse_sampled_token_logprobs(outputs)

    mock_process.assert_called_once_with(outputs, 2)
    np.testing.assert_array_equal(result, np.array([-0.1, -0.2]))


def test_parse_sampled_token_logprobs_vllm_empty():
    outputs = [MockVLLMOutput(outputs=[])]
    result = parse_sampled_token_logprobs(outputs)
    np.testing.assert_array_equal(result, np.array([]))


@patch("artefactual.preprocessing.parser.sampled_tokens_logprobs_chat_completion_api")
def test_parse_sampled_token_logprobs_openai_chat_completion(mock_process):
    mock_process.return_value = np.array([-0.3])
    outputs = MockOpenAIChatCompletion(choices=[1])

    result = parse_sampled_token_logprobs(outputs)

    mock_process.assert_called_once_with(outputs)
    np.testing.assert_array_equal(result, np.array([-0.3]))


@patch("artefactual.preprocessing.parser.is_openai_responses_api")
@patch("artefactual.preprocessing.parser.sampled_tokens_logprobs_responses_api")
def test_parse_sampled_token_logprobs_openai_responses_api(mock_process, mock_is_responses):
    mock_is_responses.return_value = True
    mock_process.return_value = np.array([-0.4])
    outputs = MockOpenAIResponsesAPI()

    result = parse_sampled_token_logprobs(outputs)

    mock_process.assert_called_once_with(outputs)
    np.testing.assert_array_equal(result, np.array([-0.4]))


def test_parse_sampled_token_logprobs_unsupported():
    with pytest.raises(TypeError, match="Unsupported output format"):
        parse_sampled_token_logprobs("unsupported_format")
