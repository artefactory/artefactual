import numpy as np
import pytest

from artefactual.scoring import EntropyContributionsMixin


def test_compute_entropy_contributions_basic():
    """Test basic entropy calculation with simple numpy array input."""
    # p = 0.5, log(p) = -0.693147
    # s = -p * log(p) = 0.346573
    logprobs = np.array([[-0.69314718, -0.69314718]], dtype=np.float32)

    result = EntropyContributionsMixin.entropy_contributions(logprobs)

    expected_entropy = -0.5 * np.log(0.5)  # ≈ 0.346573
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[expected_entropy, expected_entropy]], rtol=1e-5)


def test_compute_entropy_contributions_descending_sort():
    """Test that unsorted input produces the same result as sorted input."""
    logprobs_sorted = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
    logprobs_unsorted = np.array([[-3.0, -1.0, -2.0]], dtype=np.float32)

    result_sorted = EntropyContributionsMixin.entropy_contributions(logprobs_sorted)
    result_unsorted = EntropyContributionsMixin.entropy_contributions(logprobs_unsorted)

    np.testing.assert_allclose(result_sorted, result_unsorted, rtol=1e-5)


def test_compute_entropy_contributions_zero_logprob():
    """Test that logprob=0 (probability=1) contributes 0 entropy."""
    logprobs = np.array([[0.0]], dtype=np.float32)

    result = EntropyContributionsMixin.entropy_contributions(logprobs)

    assert result[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_compute_entropy_contributions_empty_input():
    """Test handling of empty input."""
    logprobs = np.empty((0, 0), dtype=np.float32)

    result = EntropyContributionsMixin.entropy_contributions(logprobs)

    assert result.shape == (0, 0)


def test_compute_entropy_contributions_real_data_sample():
    """Test with a sample of real data values extracted from the notebook output."""
    # Extracted from the user provided RequestOutput example
    # Token 1: ' Paris'
    # Top 3 logprobs: -1.282855, -1.345355, -1.845355

    logprobs = np.array([[-1.2828553915023804, -1.3453553915023804, -1.8453553915023804]])

    result = EntropyContributionsMixin.entropy_contributions(logprobs)

    p1 = np.exp(-1.2828553915023804)
    p2 = np.exp(-1.3453553915023804)
    p3 = np.exp(-1.8453553915023804)

    s1 = -p1 * np.log(p1)
    s2 = -p2 * np.log(p2)
    s3 = -p3 * np.log(p3)

    assert np.isclose(result[0, 0], s1, rtol=1e-5)
    assert np.isclose(result[0, 1], s2, rtol=1e-5)
    assert np.isclose(result[0, 2], s3, rtol=1e-5)
