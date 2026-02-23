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
    assert scores[0] == 0.0


def test_compute_epr_matches_token_means(mock_calibration):
    """Test sequence-level EPR equals the mean of token-level scores."""
    _ = mock_calibration
    detector = EPR("dummy", k=3)

    logprobs_seqs = [
        [{"A": -0.5, "B": -1.5}, {"C": -0.3, "D": -1.2}],
        [{"E": -1.0, "F": -2.0}, {"G": -0.8, "H": -1.8}],
    ]
    parsed_logprobs = create_parsed_logprobs(logprobs_seqs)

    seq_scores = detector.compute(parsed_logprobs)
    token_scores = detector.compute_token_scores(parsed_logprobs)

    assert len(seq_scores) == 2
    assert len(token_scores) == 2
    for seq_score, token_score in zip(seq_scores, token_scores, strict=True):
        assert np.isclose(seq_score, float(np.mean(token_score)), atol=1e-4)


def test_compute_epr_high_vs_low_confidence(mock_calibration):
    """Test that EPR distinguishes between high and low confidence outputs."""
    _ = mock_calibration
    detector = EPR("dummy", k=5)

    # High confidence: one dominant probability
    high_conf_logprobs = [
        {"A": -0.01, "B": -5.0, "C": -5.0, "D": -5.0, "E": -5.0},
        {"F": -0.01, "G": -5.0, "H": -5.0, "I": -5.0, "J": -5.0},
    ]

    # Low confidence: more uniform distribution
    low_conf_logprobs = [
        {"A": -1.7, "B": -1.7, "C": -1.7, "D": -1.7, "E": -1.7},
        {"F": -1.7, "G": -1.7, "H": -1.7, "I": -1.7, "J": -1.7},
    ]

    parsed_logprobs = create_parsed_logprobs([high_conf_logprobs, low_conf_logprobs])
    scores = detector.compute(parsed_logprobs)
    # Low confidence should have higher EPR score
    assert scores[1] > scores[0]


def test_numerical_stability(mock_calibration):
    """Test numerical stability with extreme values."""
    _ = mock_calibration
    detector = EPR("dummy", k=5)

    # Test with very small and very large (negative) logprobs
    extreme_logprobs = [
        [{"A": -0.00001, "B": -0.001, "C": -50.0, "D": -100.0, "E": -500.0}],
    ]

    parsed_logprobs = create_parsed_logprobs(extreme_logprobs)
    scores = detector.compute(parsed_logprobs)

    # Should produce valid score, not NaN or inf
    assert not np.isnan(scores[0])
    assert not np.isinf(scores[0])
    assert scores[0] >= 0


def test_compute_epr_multiple_completions_in_one_request(mock_calibration):
    """Test EPR computation when a single request has multiple completions (n > 1)."""
    _ = mock_calibration
    detector = EPR("dummy", k=3)

    logprobs_seqs = [
        [{"A": -0.5, "B": -1.5}],
        [{"C": -0.3, "D": -1.2}],
    ]
    parsed_logprobs = create_parsed_logprobs(logprobs_seqs)

    # Should return scores for BOTH completions
    seq_scores = detector.compute(parsed_logprobs)
    token_scores = detector.compute_token_scores(parsed_logprobs)

    # Verify we get results for both completions
    assert len(seq_scores) == 2, "Should return EPR for both completions"
    assert len(token_scores) == 2, "Should return token scores for both completions"

    # Both scores should be positive (valid EPR)
    assert seq_scores[0] > 0, "First completion EPR should be positive"
    assert seq_scores[1] > 0, "Second completion EPR should be positive"

    # Each completion should have token-level scores
    assert len(token_scores[0]) > 0, "First completion should have token scores"
    assert len(token_scores[1]) > 0, "Second completion should have token scores"
