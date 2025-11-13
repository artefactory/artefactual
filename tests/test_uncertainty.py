"""Unit tests for the UncertaintyDetector class."""

from dataclasses import dataclass

import numpy as np
import pytest

from artefactual.scoring.uncertainty import UncertaintyDetector

# ============================================================================
# Test Fixtures and Mock Objects
# ============================================================================


@dataclass
class MockLogprob:
    """Mock logprob object that mimics vLLM's Logprob structure."""

    logprob: float


class MockCompletionOutput:
    """Mock completion output that mimics vLLM's CompletionOutput."""

    def __init__(self, logprobs):
        self.logprobs = logprobs


class MockRequestOutput:
    """Mock request output that mimics vLLM's RequestOutput."""

    def __init__(self, outputs):
        self.outputs = outputs


def create_mock_output(token_logprobs: list[dict[str, float]]) -> MockRequestOutput:
    """Create a mock RequestOutput from token logprobs.

    Args:
        token_logprobs: List of dicts mapping tokens to logprob values

    Returns:
        MockRequestOutput object
    """
    # Convert float values to MockLogprob objects
    token_logprobs_with_objects = []
    for lp_dict in token_logprobs:
        obj_dict = {token: MockLogprob(lp) for token, lp in lp_dict.items()}
        token_logprobs_with_objects.append(obj_dict)

    completion = MockCompletionOutput(logprobs=token_logprobs_with_objects)
    return MockRequestOutput(outputs=[completion])


# ============================================================================
# Test Initialization
# ============================================================================


def test_init_default():
    """Test initialization with default parameters."""
    detector = UncertaintyDetector()
    assert detector.K == 15


def test_init_custom_k():
    """Test initialization with custom K value."""
    detector = UncertaintyDetector(K=10)
    assert detector.K == 10


def test_init_invalid_k():
    """Test that initialization fails with invalid K values."""
    with pytest.raises(ValueError, match="K must be positive"):
        UncertaintyDetector(K=0)

    with pytest.raises(ValueError, match="K must be positive"):
        UncertaintyDetector(K=-5)


# ============================================================================
# Test Entropy Contributions
# ============================================================================


def test_entropy_contributions_empty():
    """Test entropy contributions with empty input."""
    detector = UncertaintyDetector(K=5)
    result = detector._entropy_contributions([])
    assert result.shape == (1, 5)
    assert np.allclose(result, 0.0)


def test_entropy_contributions_single_token():
    """Test entropy contributions with a single token."""
    detector = UncertaintyDetector(K=3)

    # Create logprobs for one token with 3 candidates
    # Using probabilities that should give known entropy values
    logprobs = [{"A": -0.5, "B": -1.5, "C": -2.5}]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (1, 3)
    # Check that all values are non-negative (entropy contributions)
    assert np.all(result >= 0)


def test_entropy_contributions_uniform_distribution():
    """Test entropy contributions with uniform distribution (max entropy)."""
    detector = UncertaintyDetector(K=4)

    # Uniform distribution: all probabilities equal
    # log(1/4) = -1.386 (natural log)
    uniform_logprob = -1.386
    logprobs = [{"A": uniform_logprob, "B": uniform_logprob, "C": uniform_logprob, "D": uniform_logprob}]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (1, 4)
    # For uniform distribution, entropy should be maximal
    # and all contributions should be equal
    assert np.allclose(result[0, 0], result[0, 1], rtol=0.1)


def test_entropy_contributions_deterministic():
    """Test entropy contributions with deterministic distribution (min entropy)."""
    detector = UncertaintyDetector(K=3)

    # Deterministic: one probability near 1, others near 0
    logprobs = [{"A": -0.001, "B": -10.0, "C": -10.0}]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (1, 3)
    # For deterministic distribution, entropy should be minimal
    # First contribution should be very small (p ≈ 1, -p log p ≈ 0)
    assert result[0, 0] < 0.1


def test_entropy_contributions_padding():
    """Test that contributions are padded when fewer than K candidates."""
    detector = UncertaintyDetector(K=10)

    # Only 3 candidates, should be padded to 10
    logprobs = [{"A": -0.5, "B": -1.5, "C": -2.5}]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (1, 10)
    # Check that padded values are zero
    assert np.allclose(result[0, 3:], 0.0)


def test_entropy_contributions_truncation():
    """Test that contributions are truncated when more than K candidates."""
    detector = UncertaintyDetector(K=3)

    # 5 candidates, should be truncated to 3
    logprobs = [{"A": -0.5, "B": -1.0, "C": -1.5, "D": -2.0, "E": -2.5}]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (1, 3)


def test_entropy_contributions_multiple_tokens():
    """Test entropy contributions with multiple tokens."""
    detector = UncertaintyDetector(K=3)

    logprobs = [
        {"A": -0.5, "B": -1.5, "C": -2.5},
        {"D": -0.3, "E": -1.2, "F": -2.1},
        {"G": -0.8, "H": -1.8, "I": -2.8},
    ]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (3, 3)
    # Each row should have non-negative values
    assert np.all(result >= 0)


def test_entropy_contributions_with_logprob_objects():
    """Test entropy contributions with LogProbValue objects."""
    detector = UncertaintyDetector(K=3)

    # Use MockLogprob objects instead of floats
    logprobs = [
        {"A": MockLogprob(-0.5), "B": MockLogprob(-1.5), "C": MockLogprob(-2.5)},
    ]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (1, 3)
    assert np.all(result >= 0)


def test_entropy_contributions_empty_dict():
    """Test entropy contributions with empty dictionary in sequence."""
    detector = UncertaintyDetector(K=3)

    logprobs = [
        {"A": -0.5, "B": -1.5, "C": -2.5},
        {},  # Empty dict
        {"D": -0.3, "E": -1.2, "F": -2.1},
    ]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (3, 3)
    # Second row should be all zeros
    assert np.allclose(result[1], 0.0)


# ============================================================================
# Test EPR Computation
# ============================================================================


def test_compute_epr_single_output():
    """Test EPR computation with a single output."""
    detector = UncertaintyDetector(K=3)

    logprobs = [
        {"A": -0.5, "B": -1.5, "C": -2.5},
        {"D": -0.3, "E": -1.2, "F": -2.1},
    ]
    output = create_mock_output(logprobs)

    scores = detector.fit([output], return_tokens=False)
    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert scores[0] >= 0


def test_compute_epr_multiple_outputs():
    """Test EPR computation with multiple outputs."""
    detector = UncertaintyDetector(K=3)

    outputs = [
        create_mock_output([{"A": -0.5, "B": -1.5}, {"C": -0.3, "D": -1.2}]),
        create_mock_output([{"E": -1.0, "F": -2.0}, {"G": -0.8, "H": -1.8}]),
        create_mock_output([{"I": -0.2, "J": -1.5}, {"K": -0.5, "L": -2.0}]),
    ]

    scores = detector.fit(outputs, return_tokens=False)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    assert all(s >= 0 for s in scores)


def test_compute_epr_with_token_scores():
    """Test EPR computation with per-token scores."""
    detector = UncertaintyDetector(K=3)

    logprobs = [
        {"A": -0.5, "B": -1.5, "C": -2.5},
        {"D": -0.3, "E": -1.2, "F": -2.1},
    ]
    output = create_mock_output(logprobs)

    seq_scores, token_scores = detector.fit([output], return_tokens=True)

    assert len(seq_scores) == 1
    assert len(token_scores) == 1
    assert isinstance(seq_scores[0], float)
    assert isinstance(token_scores[0], np.ndarray)
    assert len(token_scores[0]) == 2  # Two tokens
    assert np.all(token_scores[0] >= 0)


def test_compute_epr_empty_outputs():
    """Test that EPR computation fails with empty outputs."""
    detector = UncertaintyDetector()

    with pytest.raises(ValueError, match="outputs cannot be empty"):
        detector.fit([])


def test_compute_epr_output_without_completions():
    """Test EPR computation with output that has no completions."""
    detector = UncertaintyDetector(K=3)

    # Create output with empty outputs list
    output = MockRequestOutput(outputs=[])

    scores = detector.fit([output], return_tokens=False)
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_compute_epr_output_without_logprobs():
    """Test EPR computation with output that has no logprobs."""
    detector = UncertaintyDetector(K=3)

    # Create output with None logprobs
    completion = MockCompletionOutput(logprobs=None)
    output = MockRequestOutput(outputs=[completion])

    scores = detector.fit([output], return_tokens=False)
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_compute_epr_high_vs_low_confidence():
    """Test that EPR distinguishes between high and low confidence outputs."""
    detector = UncertaintyDetector(K=5)

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

    high_conf_output = create_mock_output(high_conf_logprobs)
    low_conf_output = create_mock_output(low_conf_logprobs)

    scores = detector.fit([high_conf_output, low_conf_output], return_tokens=False)

    # Low confidence should have higher EPR score
    assert scores[1] > scores[0]


# ============================================================================
# Test EPR Statistics
# ============================================================================


def test_compute_epr_stats():
    """Test computation of EPR statistics."""
    detector = UncertaintyDetector(K=3)

    outputs = [
        create_mock_output([{"A": -0.5, "B": -1.5}, {"C": -0.3, "D": -1.2}]),
        create_mock_output([{"E": -1.0, "F": -2.0}, {"G": -0.8, "H": -1.8}]),
        create_mock_output([{"I": -0.2, "J": -1.5}, {"K": -0.5, "L": -2.0}]),
    ]

    stats = detector.compute_epr_stats(outputs)

    # Check that all required keys are present
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "median" in stats
    assert "q25" in stats
    assert "q75" in stats

    # Check that all values are floats
    assert all(isinstance(v, float) for v in stats.values())

    # Check that values make sense
    assert stats["min"] <= stats["q25"] <= stats["median"] <= stats["q75"] <= stats["max"]
    assert stats["std"] >= 0


def test_compute_epr_stats_empty():
    """Test that statistics computation fails with empty outputs."""
    detector = UncertaintyDetector()

    with pytest.raises(ValueError, match="outputs cannot be empty"):
        detector.compute_epr_stats([])


def test_compute_epr_stats_single_output():
    """Test statistics computation with a single output."""
    detector = UncertaintyDetector(K=3)

    output = create_mock_output([{"A": -0.5, "B": -1.5}, {"C": -0.3, "D": -1.2}])

    stats = detector.compute_epr_stats([output])

    # With single sample, many stats should be equal
    assert stats["mean"] == stats["median"]
    assert stats["min"] == stats["max"]
    assert stats["std"] == 0.0


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_entropy_contributions_zero_probabilities():
    """Test that zero probabilities are handled correctly."""
    detector = UncertaintyDetector(K=3)

    # Extreme case: very low probabilities
    logprobs = [{"A": -0.001, "B": -100.0, "C": -100.0}]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (1, 3)
    # Should not produce NaN or inf
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_entropy_contributions_mixed_types():
    """Test entropy contributions with mixed float and LogProbValue objects."""
    detector = UncertaintyDetector(K=3)

    # Mix floats and MockLogprob objects
    logprobs = [
        {"A": -0.5, "B": MockLogprob(-1.5), "C": -2.5},
    ]

    result = detector._entropy_contributions(logprobs)
    assert result.shape == (1, 3)
    assert np.all(result >= 0)


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    detector = UncertaintyDetector(K=5)

    # Test with very small and very large (negative) logprobs
    extreme_logprobs = [
        {"A": -0.00001, "B": -0.001, "C": -50.0, "D": -100.0, "E": -500.0},
    ]

    output = create_mock_output(extreme_logprobs)
    scores = detector.fit([output], return_tokens=False)

    # Should produce valid score, not NaN or inf
    assert not np.isnan(scores[0])
    assert not np.isinf(scores[0])
    assert scores[0] >= 0


def test_compute_epr_multiple_completions_in_one_request():
    """Test EPR computation when a single request has multiple completions (n > 1)."""
    detector = UncertaintyDetector(K=3)

    # Create a single RequestOutput with two CompletionOutputs
    # This simulates vLLM output when sampling_params.n=2
    completion1 = MockCompletionOutput(logprobs=[{"A": -0.5, "B": -1.5}])
    completion2 = MockCompletionOutput(logprobs=[{"C": -0.3, "D": -1.2}])
    output = MockRequestOutput(outputs=[completion1, completion2])

    # Should return scores for BOTH completions, not just the first one
    seq_scores, token_scores = detector.fit([output], return_tokens=True)

    # Verify we get results for both completions
    assert len(seq_scores) == 2, "Should return EPR for both completions"
    assert len(token_scores) == 2, "Should return token scores for both completions"

    # Both scores should be positive (valid EPR)
    assert seq_scores[0] > 0, "First completion EPR should be positive"
    assert seq_scores[1] > 0, "Second completion EPR should be positive"

    # Each completion should have token-level scores
    assert len(token_scores[0]) > 0, "First completion should have token scores"
    assert len(token_scores[1]) > 0, "Second completion should have token scores"
