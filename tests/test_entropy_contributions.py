import numpy as np

from artefactual.scoring.entropy_methods.entropy_contributions import compute_entropy_contributions


class MockLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


def test_compute_entropy_contributions_basic():
    """Test basic entropy calculation with simple numpy array input."""
    # p = 0.5, log(p) = -0.693147
    # s = -p * log(p) = 0.346573
    logprobs = np.array([[-0.69314718, -0.69314718]], dtype=np.float32)
    k = 2

    result = compute_entropy_contributions(logprobs, k)

    expected_entropy = -0.5 * np.log(0.5)  # ≈ 0.346573
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[expected_entropy, expected_entropy]], rtol=1e-5)


def test_compute_entropy_contributions_padding():
    """Test that output is correctly padded when input has fewer than k logprobs."""
    logprobs = np.array([[-1.0]], dtype=np.float32)
    k = 3

    result = compute_entropy_contributions(logprobs, k)

    assert result.shape == (1, 3)
    # First element should be computed entropy
    expected = -np.exp(-1.0) * -1.0
    assert np.isclose(result[0, 0], expected)
    # Remaining elements should be 0.0
    assert result[0, 1] == 0.0
    assert result[0, 2] == 0.0


def test_compute_entropy_contributions_truncation():
    """Test that output is correctly truncated when input has more than k logprobs."""
    logprobs = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
    k = 2

    result = compute_entropy_contributions(logprobs, k)

    assert result.shape == (1, 2)
    expected_0 = -np.exp(-1.0) * -1.0
    expected_1 = -np.exp(-2.0) * -2.0
    assert np.isclose(result[0, 0], expected_0)
    assert np.isclose(result[0, 1], expected_1)


def test_compute_entropy_contributions_ragged_input():
    """Test handling of ragged list input (different number of logprobs per token)."""
    # Token 1 has 1 logprob, Token 2 has 2 logprobs
    logprobs = [[-1.0], [-1.0, -2.0]]
    k = 2

    result = compute_entropy_contributions(logprobs, k)

    assert result.shape == (2, 2)

    # Check first token (padded)
    expected_0 = -np.exp(-1.0) * -1.0
    assert np.isclose(result[0, 0], expected_0)
    assert result[0, 1] == 0.0

    # Check second token
    expected_1_0 = -np.exp(-1.0) * -1.0
    expected_1_1 = -np.exp(-2.0) * -2.0
    assert np.isclose(result[1, 0], expected_1_0)
    assert np.isclose(result[1, 1], expected_1_1)


def test_compute_entropy_contributions_mock_objects():
    """Test handling of objects with .logprob attribute (simulating vLLM Logprob objects)."""
    logprobs = [[MockLogprob(-1.0), MockLogprob(-2.0)], [MockLogprob(-0.5)]]
    k = 2

    result = compute_entropy_contributions(logprobs, k)

    assert result.shape == (2, 2)

    # Token 1
    expected_0_0 = -np.exp(-1.0) * -1.0
    expected_0_1 = -np.exp(-2.0) * -2.0
    assert np.isclose(result[0, 0], expected_0_0)
    assert np.isclose(result[0, 1], expected_0_1)

    # Token 2 (padded)
    expected_1_0 = -np.exp(-0.5) * -0.5
    assert np.isclose(result[1, 0], expected_1_0)
    assert result[1, 1] == 0.0


def test_compute_entropy_contributions_empty_input():
    """Test handling of empty input."""
    logprobs = []
    k = 5

    result = compute_entropy_contributions(logprobs, k)

    assert result.shape == (0, k)


def test_compute_entropy_contributions_real_data_sample():
    """Test with a sample of real data values extracted from the notebook output."""
    # Extracted from the user provided RequestOutput example
    # Token 1: ' Paris'
    # Top 3 logprobs: -1.282855, -1.345355, -1.845355

    logprobs = [[-1.2828553915023804, -1.3453553915023804, -1.8453553915023804]]
    k = 3

    result = compute_entropy_contributions(logprobs, k)

    # Manual calculation check
    p1 = np.exp(-1.2828553915023804)
    p2 = np.exp(-1.3453553915023804)
    p3 = np.exp(-1.8453553915023804)

    s1 = -p1 * np.log(p1)
    s2 = -p2 * np.log(p2)
    s3 = -p3 * np.log(p3)

    assert np.isclose(result[0, 0], s1, rtol=1e-5)
    assert np.isclose(result[0, 1], s2, rtol=1e-5)
    assert np.isclose(result[0, 2], s3, rtol=1e-5)
