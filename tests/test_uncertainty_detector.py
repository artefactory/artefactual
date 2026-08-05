"""Integration and base-class tests for uncertainty detectors."""

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from artefactual.scoring import (
    EPR,
    LogProbUncertaintyDetector,
)


@pytest.fixture
def mock_calibration():
    with patch("artefactual.scoring.entropy_methods.epr.load_calibration") as mock:
        mock.return_value = {"intercept": 0.0, "coefficients": {"mean_entropy": 1.0}}
        yield mock


def create_parsed_logprobs(logprobs_sequences: list[list[dict[str, float | Any]]]) -> list[dict[int, list[float]]]:
    """Create parsed logprobs from a list of logprob sequences."""
    parsed_output = []
    for seq in logprobs_sequences:
        seq_dict = {}
        for i, token_logprobs in enumerate(seq):
            values = []
            for v in token_logprobs.values():
                if hasattr(v, "logprob"):
                    values.append(v.logprob)
                else:
                    values.append(v)
            seq_dict[i] = values
        parsed_output.append(seq_dict)
    return parsed_output


class ConcreteUncertaintyDetector(LogProbUncertaintyDetector):
    """Concrete implementation used to validate abstract base behavior."""

    def compute(self, inputs: Any) -> list[float]:  # noqa: ARG002
        return []

    def compute_token_scores(self, inputs: Any) -> list[NDArray[np.floating]]:  # noqa: ARG002
        return []


def test_init_default():
    """Test initialization with default parameters."""
    detector = ConcreteUncertaintyDetector()
    assert detector.k == 15


def test_init_custom_k():
    """Test initialization with custom K value."""
    detector = ConcreteUncertaintyDetector(k=10)
    assert detector.k == 10


def test_init_invalid_k():
    """Test that initialization fails with invalid K values."""
    with pytest.raises(ValueError, match="k must be positive"):
        ConcreteUncertaintyDetector(k=0)

    with pytest.raises(ValueError, match="k must be positive"):
        ConcreteUncertaintyDetector(k=-5)


def test_compute_epr_empty_outputs(mock_calibration):
    """Test that EPR computation fails with empty outputs."""
    _ = mock_calibration
    detector = EPR("dummy")

    scores = detector.compute([])
    assert scores == []


def test_compute_epr_output_without_logprobs(mock_calibration):
    """Test EPR computation with output that has no logprobs."""
    _ = mock_calibration
    detector = EPR("dummy", k=3)
    detector.is_calibrated = False

    # Create output with no logprobs
    parsed_logprobs = create_parsed_logprobs([[]])

    scores = detector.compute(parsed_logprobs)
    assert len(scores) == 1
    np.testing.assert_allclose(scores[0], 0.0, atol=1e-12)
