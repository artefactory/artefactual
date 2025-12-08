"""Unit tests for the UncertaintyDetector class."""

from dataclasses import dataclass

import numpy as np
import pytest
from vllm import RequestOutput

from artefactual.scoring.entropy_methods.entropy_contributions import compute_entropy_contributions
from artefactual.scoring.entropy_methods.epr import EPR
from artefactual.scoring.entropy_methods.uncertainty_detector import UncertaintyDetector

# ============================================================================
# Test Fixtures and Mock Objects
# ============================================================================


@dataclass
class MockLogprob:
    """Mock logprob object that mimics vLLM's Logprob structure."""

    logprob: float


@dataclass
class MockCompletionOutput:
    """Mock completion output that mimics vLLM's CompletionOutput."""

    logprobs: list[dict[str, "MockLogprob"]] | None


class MockRequestOutput(RequestOutput):
    """Mock request output that mimics vLLM's RequestOutput."""

    def __init__(self, outputs: list[MockCompletionOutput]):
        self.outputs = outputs


def create_request_output(logprobs_sequences: list[list[dict[str, float]]]) -> list[MockRequestOutput]:
    """Create a mock RequestOutput object from a list of logprob sequences."""
    mock_completion_outputs = []
    for seq in logprobs_sequences:
        if not seq:  # Handle empty sequences
            mock_completion_outputs.append(MockCompletionOutput(logprobs=None))
            continue

        mock_logprobs_list = []
        for token_logprobs in seq:
            mock_logprobs_list.append({k: MockLogprob(v) for k, v in token_logprobs.items()})
        mock_completion_outputs.append(MockCompletionOutput(logprobs=mock_logprobs_list))
    return [MockRequestOutput(outputs=mock_completion_outputs)]


# ============================================================================
# Test Initialization
# ============================================================================


def test_init_default():
    """Test initialization with default parameters."""
    detector = UncertaintyDetector()
    assert detector.k == 15


def test_init_custom_k():
    """Test initialization with custom K value."""
    detector = UncertaintyDetector(k=10)
    assert detector.k == 10


def test_init_invalid_k():
    """Test that initialization fails with invalid K values."""
    with pytest.raises(ValueError, match="k must be positive"):
        UncertaintyDetector(k=0)

    with pytest.raises(ValueError, match="k must be positive"):
        UncertaintyDetector(k=-5)


# ============================================================================
# Test Entropy Contributions
# ============================================================================


def test_entropy_contributions_empty():
    """Test entropy contributions with empty input."""
    detector = UncertaintyDetector(k=5)
    result = compute_entropy_contributions([], detector.k)
    assert result.shape == (0, 5)
    assert np.allclose(result, 0.0)


def test_entropy_contributions_single_token():
    """Test entropy contributions with a single token."""
    detector = UncertaintyDetector(k=3)

    # Create logprobs for one token with 3 candidates
    # Using probabilities that should give known entropy values
    logprobs = [{"A": -0.5, "B": -1.5, "C": -2.5}]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (1, 3)
    # Check that all values are non-negative (entropy contributions)
    assert np.all(result >= 0)


def test_entropy_contributions_uniform_distribution():
    """Test entropy contributions with uniform distribution (max entropy)."""
    detector = UncertaintyDetector(k=4)

    # Uniform distribution: all probabilities equal
    # log(1/4) = -1.386 (natural log)
    uniform_logprob = -1.386
    logprobs = [{"A": uniform_logprob, "B": uniform_logprob, "C": uniform_logprob, "D": uniform_logprob}]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (1, 4)
    # For uniform distribution, entropy should be maximal
    # and all contributions should be equal
    assert np.allclose(result[0, 0], result[0, 1], rtol=0.1)


def test_entropy_contributions_deterministic():
    """Test entropy contributions with deterministic distribution (min entropy)."""
    detector = UncertaintyDetector(k=3)

    # Deterministic: one probability near 1, others near 0
    logprobs = [{"A": -0.001, "B": -10.0, "C": -10.0}]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (1, 3)
    # For deterministic distribution, entropy should be minimal
    # First contribution should be very small (p ≈ 1, -p log p ≈ 0)
    assert result[0, 0] < 0.1


def test_entropy_contributions_padding():
    """Test that contributions are padded when fewer than K candidates."""
    detector = UncertaintyDetector(k=10)

    # Only 3 candidates, should be padded to 10
    logprobs = [{"A": -0.5, "B": -1.5, "C": -2.5}]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (1, 10)
    # Check that padded values are zero
    assert np.allclose(result[0, 3:], 0.0)


def test_entropy_contributions_truncation():
    """Test that contributions are truncated when more than K candidates."""
    detector = UncertaintyDetector(k=3)

    # 5 candidates, should be truncated to 3
    logprobs = [{"A": -0.5, "B": -1.0, "C": -1.5, "D": -2.0, "E": -2.5}]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (1, 3)


def test_entropy_contributions_multiple_tokens():
    """Test entropy contributions with multiple tokens."""
    detector = UncertaintyDetector(k=3)

    logprobs = [
        {"A": -0.5, "B": -1.5, "C": -2.5},
        {"D": -0.3, "E": -1.2, "F": -2.1},
        {"G": -0.8, "H": -1.8, "I": -2.8},
    ]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (3, 3)
    # Each row should have non-negative values
    assert np.all(result >= 0)


def test_entropy_contributions_with_logprob_objects():
    """Test entropy contributions with LogProbValue objects."""
    detector = UncertaintyDetector(k=3)

    # Use MockLogprob objects instead of floats
    logprobs = [
        {"A": MockLogprob(-0.5), "B": MockLogprob(-1.5), "C": MockLogprob(-2.5)},
    ]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (1, 3)
    assert np.all(result >= 0)


def test_entropy_contributions_empty_dict():
    """Test entropy contributions with empty dictionary in sequence."""
    detector = UncertaintyDetector(k=3)

    logprobs = [
        {"A": -0.5, "B": -1.5, "C": -2.5},
        {},  # Empty dict
        {"D": -0.3, "E": -1.2, "F": -2.1},
    ]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (3, 3)
    # Second row should be all zeros
    assert np.allclose(result[1], 0.0)


# ============================================================================
# Test EPR Computation
# ============================================================================


def test_compute_epr_single_output():
    """Test EPR computation with a single output."""
    detector = EPR(k=3)

    logprobs_seq = [
        [
            {"A": -0.5, "B": -1.5, "C": -2.5},
            {"D": -0.3, "E": -1.2, "F": -2.1},
        ]
    ]
    request_output = create_request_output(logprobs_seq)

    scores = detector.compute(request_output, return_per_token_scores=False)
    assert len(scores) == 1
    assert isinstance(scores[0], float)
    assert scores[0] >= 0


def test_compute_epr_multiple_outputs():
    """Test EPR computation with multiple outputs."""
    detector = EPR(k=3)

    logprobs_seqs = [
        [{"A": -0.5, "B": -1.5}, {"C": -0.3, "D": -1.2}],
        [{"E": -1.0, "F": -2.0}, {"G": -0.8, "H": -1.8}],
        [{"I": -0.2, "J": -1.5}, {"K": -0.5, "L": -2.0}],
    ]
    request_output = create_request_output(logprobs_seqs)

    scores = detector.compute(request_output, return_per_token_scores=False)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    assert all(s >= 0 for s in scores)


def test_compute_epr_with_token_scores():
    """Test EPR computation with per-token scores."""
    detector = EPR(k=3)

    logprobs_seq = [
        [
            {"A": -0.5, "B": -1.5, "C": -2.5},
            {"D": -0.3, "E": -1.2, "F": -2.1},
        ]
    ]
    request_output = create_request_output(logprobs_seq)

    seq_scores, token_scores = detector.compute(request_output, return_per_token_scores=True)

    assert len(seq_scores) == 1
    assert len(token_scores) == 1
    assert isinstance(seq_scores[0], float)
    assert isinstance(token_scores[0], np.ndarray)
    assert len(token_scores[0]) == 2  # Two tokens
    assert np.all(token_scores[0] >= 0)


def test_compute_epr_empty_outputs():
    """Test that EPR computation fails with empty outputs."""
    detector = EPR()

    scores = detector.compute([])
    assert scores == []


def test_compute_epr_output_without_completions():
    """Test EPR computation with output that has no completions."""
    detector = EPR(k=3)

    # Empty list of completions
    scores = detector.compute([], return_per_token_scores=False)
    assert len(scores) == 0


def test_compute_epr_output_without_logprobs():
    """Test EPR computation with output that has no logprobs."""
    detector = EPR(k=3)

    # Create output with no logprobs
    request_output = create_request_output([[]])

    scores = detector.compute(request_output, return_per_token_scores=False)
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_compute_epr_high_vs_low_confidence():
    """Test that EPR distinguishes between high and low confidence outputs."""
    detector = EPR(k=5)

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

    request_output = create_request_output([high_conf_logprobs, low_conf_logprobs])
    scores = detector.compute(request_output, return_per_token_scores=False)
    # Low confidence should have higher EPR score
    assert scores[1] > scores[0]


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_entropy_contributions_zero_probabilities():
    """Test that zero probabilities are handled correctly."""
    detector = UncertaintyDetector(k=3)

    # Extreme case: very low probabilities
    logprobs = [{"A": -0.001, "B": -100.0, "C": -100.0}]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (1, 3)
    # Should not produce NaN or inf
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_entropy_contributions_mixed_types():
    """Test entropy contributions with mixed float and LogProbValue objects."""
    detector = UncertaintyDetector(k=3)

    # Mix floats and MockLogprob objects
    logprobs = [
        {"A": -0.5, "B": MockLogprob(-1.5), "C": -2.5},
    ]

    result = compute_entropy_contributions(logprobs, detector.k)
    assert result.shape == (1, 3)
    assert np.all(result >= 0)


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    detector = EPR(k=5)

    # Test with very small and very large (negative) logprobs
    extreme_logprobs = [
        [{"A": -0.00001, "B": -0.001, "C": -50.0, "D": -100.0, "E": -500.0}],
    ]

    request_output = create_request_output(extreme_logprobs)
    scores = detector.compute(request_output, return_per_token_scores=False)

    # Should produce valid score, not NaN or inf
    assert not np.isnan(scores[0])
    assert not np.isinf(scores[0])
    assert scores[0] >= 0


def test_compute_epr_multiple_completions_in_one_request():
    """Test EPR computation when a single request has multiple completions (n > 1)."""
    detector = EPR(k=3)

    logprobs_seqs = [
        [{"A": -0.5, "B": -1.5}],
        [{"C": -0.3, "D": -1.2}],
    ]
    request_output = create_request_output(logprobs_seqs)

    # Should return scores for BOTH completions
    seq_scores, token_scores = detector.compute(request_output, return_per_token_scores=True)

    # Verify we get results for both completions
    assert len(seq_scores) == 2, "Should return EPR for both completions"
    assert len(token_scores) == 2, "Should return token scores for both completions"

    # Both scores should be positive (valid EPR)
    assert seq_scores[0] > 0, "First completion EPR should be positive"
    assert seq_scores[1] > 0, "Second completion EPR should be positive"

    # Each completion should have token-level scores
    assert len(token_scores[0]) > 0, "First completion should have token scores"
    assert len(token_scores[1]) > 0, "Second completion should have token scores"
