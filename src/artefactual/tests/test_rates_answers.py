import json
from typing import Literal
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from artefactual.calibration.rates_answers import (
    RatingConfig,
    ResultItem,
    _extract_answer_text,  # noqa: PLC2701
    _parse_judgment,  # noqa: PLC2701
    _prepare_judgment_messages,  # noqa: PLC2701
    rate_answers,
)


# Test _parse_judgment
@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("True", True),
        ("true", True),
        ("TRUE", True),
        ("True.", True),
        ("False", False),
        ("false", False),
        ("FALSE", False),
        ("False.", False),
        ("None", None),
        ("random text", None),
        ("  True  ", True),
    ],
)
def test_parse_judgment(
    input_text: Literal["True"]
    | Literal["true"]
    | Literal["TRUE"]
    | Literal["True."]
    | Literal["False"]
    | Literal["false"]
    | Literal["FALSE"]
    | Literal["False."]
    | Literal["None"]
    | Literal["random text"]
    | Literal["  True  "],
    *,
    expected: bool | None,
):
    assert _parse_judgment(input_text) == expected


# Test _extract_answer_text
def test_extract_answer_text():
    # Case 1: Digit key
    assert _extract_answer_text({"0": "answer", "epr_score": 0.5}) == "answer"
    # Case 2: No digit key, fallback
    assert _extract_answer_text({"text": "answer", "epr_score": 0.5}) == "answer"
    # Case 3: Only epr_score
    assert _extract_answer_text({"epr_score": 0.5}) is None
    # Case 4: Multiple keys, digit priority
    assert _extract_answer_text({"0": "digit_ans", "other": "other_ans", "epr_score": 0.5}) == "digit_ans"


# Test ResultItem
def test_result_item_validation():
    # Valid item
    data = {
        "query_id": "q1",
        "query": "test query",
        "expected_answers": ["ans1"],
        "generated_answers": [{"0": "gen1", "epr_score": 0.1}],
    }
    item = ResultItem(**data)
    assert item.query_id == "q1"
    assert item.get_expected_answers() == ["ans1"]

    # Valid item with string expected_answer
    data_str = data.copy()
    data_str["expected_answers"] = "ans1"
    item = ResultItem(**data_str)
    assert item.get_expected_answers() == ["ans1"]

    # Valid item with expected_answer key (singular)
    data_singular = {"query_id": "q1", "query": "test query", "expected_answer": "ans1", "generated_answers": []}
    item = ResultItem(**data_singular)
    assert item.get_expected_answers() == ["ans1"]

    # Missing expected answers
    data_missing = {"query_id": "q1", "query": "test query", "generated_answers": []}
    item = ResultItem(**data_missing)
    assert item.get_expected_answers() == []


# Test _prepare_judgment_messages
def test_prepare_judgment_messages():
    msgs = _prepare_judgment_messages("q", ["exp"], "gen")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "Question: q" in msgs[0]["content"]
    assert "Expected Answer(s): ['exp']" in msgs[0]["content"]
    assert "Generated Answer: gen" in msgs[0]["content"]


# Test rate_answers
@patch("artefactual.calibration.rates_answer.init_llm")
def test_rate_answers(mock_init_llm):
    # Mock LLM
    mock_llm = MagicMock()
    mock_init_llm.return_value = mock_llm

    # Mock LLM output
    # We expect 2 calls to chat (actually 1 call with list of messages)
    # The function calls llm.chat(messages=messages_list, ...)

    mock_output_true = MagicMock()
    mock_output_true.outputs = [MagicMock(text="True")]

    mock_output_false = MagicMock()
    mock_output_false.outputs = [MagicMock(text="False")]

    # llm.chat returns a list of RequestOutput objects
    mock_llm.chat.return_value = [mock_output_true, mock_output_false]

    # Mock input file content
    input_data = {
        "results": [
            {
                "query_id": "q1",
                "query": "query1",
                "expected_answers": ["exp1"],
                "generated_answers": [{"0": "gen1", "epr_score": 0.1}],
            },
            {
                "query_id": "q2",
                "query": "query2",
                "expected_answers": ["exp2"],
                "generated_answers": [{"0": "gen2", "epr_score": 0.9}],
            },
        ]
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(input_data))):
        config = RatingConfig(input_file="dummy.json")
        df = rate_answers(config)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "judgment" in df.columns
    assert "uncertainty_score" in df.columns

    # Check values
    # q1 -> True
    assert bool(df.loc["q1", "judgment"]) is True
    assert df.loc["q1", "uncertainty_score"] == 0.1

    # q2 -> False
    assert bool(df.loc["q2", "judgment"]) is False
    assert df.loc["q2", "uncertainty_score"] == 0.9


def test_rate_answers_file_not_found():
    config = RatingConfig(input_file="non_existent.json")
    with patch("builtins.open", side_effect=FileNotFoundError):
        df = rate_answers(config)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_rate_answers_empty_results():
    input_data = {"results": []}
    with patch("builtins.open", mock_open(read_data=json.dumps(input_data))):
        config = RatingConfig(input_file="dummy.json")
        df = rate_answers(config)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


@patch("artefactual.calibration.rates_answer.init_llm")
def test_rate_answers_invalid_items_skipped(mock_init_llm):
    # Mock LLM
    mock_llm = MagicMock()
    mock_init_llm.return_value = mock_llm
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="True")]
    mock_llm.chat.return_value = [mock_output]

    input_data = {
        "results": [
            {
                "query_id": "q1",
                "query": "query1",
                "expected_answers": ["exp1"],
                "generated_answers": [{"0": "gen1", "epr_score": 0.1}],
            },
            {
                "query_id": "q2",
                # Missing query
                "expected_answers": ["exp2"],
                "generated_answers": [{"0": "gen2", "epr_score": 0.9}],
            },
        ]
    }

    with patch("builtins.open", mock_open(read_data=json.dumps(input_data))):
        config = RatingConfig(input_file="dummy.json")
        df = rate_answers(config)

    # Only q1 should be processed
    assert len(df) == 1
    assert df.index[0] == "q1"
