from unittest.mock import patch

import numpy as np
import pytest

from artefactual.scoring import EPR

# Calibration data from src/artefactual/data/calibration_ministral.json
CALIBRATION_DATA = {"intercept": -2.9149738672340084, "coefficients": {"mean_entropy": 58.16536593597155}}


@pytest.fixture
def mock_load_calibration():
    with patch("artefactual.scoring.entropy_methods.epr.load_calibration") as mock:
        mock.return_value = CALIBRATION_DATA
        yield mock


def test_epr_initialization_uncalibrated():
    with pytest.raises(ValueError, match="Failed to load calibration"):
        EPR()


def test_epr_initialization_calibrated(mock_load_calibration):
    epr = EPR(pretrained_model_name_or_path="test_model")
    assert epr.is_calibrated
    assert epr.intercept == CALIBRATION_DATA["intercept"]
    assert epr.coefficient == CALIBRATION_DATA["coefficients"]["mean_entropy"]
    mock_load_calibration.assert_called_once_with("test_model")


@patch("artefactual.scoring.entropy_methods.epr.load_calibration")
def test_epr_compute_uncalibrated(mock_load):
    # Mock identity calibration
    mock_load.return_value = {"intercept": 0.0, "coefficients": {"mean_entropy": 1.0}}
    epr = EPR("dummy")
    epr.is_calibrated = False  # Force uncalibrated to test raw entropy

    # Mock output: 1 completion, 1 token, 1 logprob
    # logprob = -1.0 -> p = 0.3678 -> s = -0.3678 * -1.0 = 0.3678
    mock_parsed = [{0: [-1.0]}]  # 1 token at index 0 with logprob -1.0

    with patch.object(EPR, "_parse_outputs", return_value=mock_parsed):
        scores = epr.compute("dummy_input")

    assert len(scores) == 1
    expected_entropy = -np.exp(-1.0) * -1.0  # ≈ 0.367879
    assert np.isclose(scores[0], expected_entropy, rtol=1e-5)


def test_epr_compute_calibrated(mock_load_calibration):
    epr = EPR(pretrained_model_name_or_path="test_model")
    mock_load_calibration.assert_called_with("test_model")

    # Mock output: 1 completion, 1 token, 1 logprob
    # logprob = -1.0 -> entropy ≈ 0.367879
    mock_parsed = [{0: [-1.0]}]

    with patch.object(EPR, "_parse_outputs", return_value=mock_parsed):
        scores = epr.compute("dummy_input")

    assert len(scores) == 1

    raw_entropy = -np.exp(-1.0) * -1.0
    intercept = CALIBRATION_DATA["intercept"]
    coeff = CALIBRATION_DATA["coefficients"]["mean_entropy"]

    linear_score = coeff * raw_entropy + intercept
    expected_score = 1.0 / (1.0 + np.exp(-linear_score))

    assert np.isclose(scores[0], expected_score, rtol=1e-5)


def test_epr_compute_token_scores_calibrated(mock_load_calibration):
    epr = EPR(pretrained_model_name_or_path="test_model")
    mock_load_calibration.assert_called_with("test_model")

    # Mock output: 1 completion, 2 tokens
    # Token 0: logprob -1.0 -> entropy ≈ 0.367879
    # Token 1: logprob -0.5 -> entropy ≈ 0.303265
    mock_parsed = [{0: [-1.0], 1: [-0.5]}]

    with patch.object(EPR, "_parse_outputs", return_value=mock_parsed):
        token_scores = epr.compute_token_scores("dummy_input")

    assert len(token_scores) == 1
    assert len(token_scores[0]) == 2

    intercept = CALIBRATION_DATA["intercept"]
    coeff = CALIBRATION_DATA["coefficients"]["mean_entropy"]

    raw_entropy_0 = -np.exp(-1.0) * -1.0
    raw_entropy_1 = -np.exp(-0.5) * -0.5

    expected_0 = 1.0 / (1.0 + np.exp(-(coeff * raw_entropy_0 + intercept)))
    expected_1 = 1.0 / (1.0 + np.exp(-(coeff * raw_entropy_1 + intercept)))

    assert np.isclose(token_scores[0][0], expected_0, rtol=1e-5)
    assert np.isclose(token_scores[0][1], expected_1, rtol=1e-5)


def test_epr_empty_completion_calibrated(mock_load_calibration):
    epr = EPR(pretrained_model_name_or_path="test_model")
    mock_load_calibration.assert_called_with("test_model")

    # Mock output: 1 completion, 0 tokens
    mock_parsed = [{}]

    with patch.object(EPR, "_parse_outputs", return_value=mock_parsed):
        scores = epr.compute("dummy_input")

    assert len(scores) == 1

    # Should return sigmoid(intercept)
    intercept = CALIBRATION_DATA["intercept"]
    expected_score = 1.0 / (1.0 + np.exp(-intercept))

    assert np.isclose(scores[0], expected_score, rtol=1e-5)


@patch("artefactual.scoring.entropy_methods.epr.load_calibration")
def test_epr_compute_specific_values_from_notebook(mock_load):
    """
    Test EPR computation with specific values from a vLLM example.
    Verifies exact entropy values for 2 tokens and their mean.
    """
    # Mock identity calibration
    mock_load.return_value = {"intercept": 0.0, "coefficients": {"mean_entropy": 1.0}}
    epr = EPR("dummy", k=15)  # Ensure k=15 matches the data
    epr.is_calibrated = False  # Force uncalibrated to test raw entropy

    # Data from notebook
    token1_logprobs = [
        -1.2828553915023804,
        -1.3453553915023804,
        -1.8453553915023804,
        -2.09535551071167,
        -4.22035551071167,
        -4.40785551071167,
        -4.97035551071167,
        -5.37660551071167,
        -5.40785551071167,
        -5.50160551071167,
        -5.53285551071167,
        -5.56410551071167,
        -5.75160551071167,
        -5.78285551071167,
        -5.81410551071167,
    ]

    token2_logprobs = [
        -0.9893605709075928,
        -1.1143605709075928,
        -1.4893605709075928,
        -3.4893605709075928,
        -3.6143605709075928,
        -5.239360809326172,
        -5.489360809326172,
        -6.114360809326172,
        -6.676860809326172,
        -7.176860809326172,
        -8.176860809326172,
        -8.676860809326172,
        -8.801860809326172,
        -8.801860809326172,
        -8.989360809326172,
    ]

    # Mock parsed output: 1 completion, 2 tokens
    mock_parsed = [{0: token1_logprobs, 1: token2_logprobs}]

    with patch.object(EPR, "_parse_outputs", return_value=mock_parsed):
        # Check token scores
        token_scores = epr.compute_token_scores("dummy_input")
        assert len(token_scores) == 1
        assert len(token_scores[0]) == 2

        # Expected values provided by user
        expected_token1 = 1.5737461103163657
        expected_token2 = 1.3586518373670613

        assert np.isclose(token_scores[0][0], expected_token1, rtol=1e-5)
        assert np.isclose(token_scores[0][1], expected_token2, rtol=1e-5)

        # Check sequence score (mean)
        seq_scores = epr.compute("dummy_input")
        assert len(seq_scores) == 1

        expected_seq = 1.4661989738417134
        assert np.isclose(seq_scores[0], expected_seq, rtol=1e-5)


def test_epr_initialization_failure():
    with patch(
        "artefactual.scoring.entropy_methods.epr.load_calibration", side_effect=ValueError("Calibration not found")
    ):
        with pytest.raises(ValueError, match="Calibration not found"):
            EPR(pretrained_model_name_or_path="invalid_model")
