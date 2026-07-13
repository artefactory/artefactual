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


def test_epr_initialization_no_path():
    with pytest.warns(UserWarning, match="EPR is currently not calibrated"):
        epr = EPR()
        assert not epr.is_calibrated
        assert epr.intercept == 0.0
        assert epr.coefficient == 1.0


def test_epr_initialization_calibrated(mock_load_calibration):
    epr = EPR(pretrained_model_name_or_path="test_model")
    assert epr.is_calibrated
    assert epr.intercept == CALIBRATION_DATA["intercept"]
    assert epr.coefficient == CALIBRATION_DATA["coefficients"]["mean_entropy"]
    mock_load_calibration.assert_called_once_with("test_model")


def test_epr_empty_completion_calibrated(mock_load_calibration):
    epr = EPR(pretrained_model_name_or_path="test_model")
    mock_load_calibration.assert_called_with("test_model")

    # Mock output: 1 completion, 0 tokens
    mock_parsed = [{}]

    scores = epr.compute(mock_parsed)

    assert len(scores) == 1

    # Should return sigmoid(intercept)
    intercept = CALIBRATION_DATA["intercept"]
    expected_score = 1.0 / (1.0 + np.exp(-intercept))

    assert np.isclose(scores[0], expected_score, rtol=1e-5)


def test_epr_initialization_failure():
    with (
        patch(
            "artefactual.scoring.entropy_methods.epr.load_calibration", side_effect=ValueError("Calibration not found")
        ),
        pytest.raises(ValueError, match="Calibration not found"),
    ):
        EPR(pretrained_model_name_or_path="invalid_model")
