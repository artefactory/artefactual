"""Tests for the entropy.calculations module."""

import numpy as np
from absl.testing import absltest, parameterized
from artefactual.entropy.calculations import (
    entropy_from_logprobs,
    mean_entropy,
    token_entropy,
)
from hypothesis import given
from hypothesis import strategies as st


class EntropyCalculationsTest(parameterized.TestCase):
    """Test cases for the entropy.calculations module."""

    def test_entropy_from_logprobs_single(self):
        """Test entropy calculation for a single distribution."""
        # Uniform distribution with 2 tokens
        logprobs = np.array([-0.693, -0.693])  # log(0.5), log(0.5)
        entropy = entropy_from_logprobs(logprobs)
        # Expected entropy for uniform distribution: log(n) = log(2) = 0.693
        self.assertAlmostEqual(entropy, 0.693, places=3)

    def test_entropy_from_logprobs_non_uniform(self):
        """Test entropy calculation for a non-uniform distribution."""
        # Non-uniform distribution: [0.8, 0.2]
        logprobs = np.array([-0.223, -1.609])  # log(0.8), log(0.2)
        entropy = entropy_from_logprobs(logprobs)
        # Expected entropy: -0.8*log(0.8) - 0.2*log(0.2) ≈ 0.5
        self.assertAlmostEqual(entropy, 0.5, places=1)

    def test_entropy_from_logprobs_deterministic(self):
        """Test entropy calculation for a deterministic distribution."""
        # Deterministic distribution: [1.0, 0.0]
        # Use a very negative number instead of -inf
        logprobs = np.array([0.0, -30.0])  # log(1.0), log(~0.0)
        entropy = entropy_from_logprobs(logprobs)
        # Expected entropy: nearly 0
        self.assertAlmostEqual(entropy, 0.0, places=5)

    def test_entropy_from_logprobs_rejects_inf(self):
        """Test that an error is raised for inf values."""
        logprobs = np.array([0.0, -float("inf")])
        with self.assertRaises(ValueError):
            entropy_from_logprobs(logprobs)

    def test_entropy_from_logprobs_empty(self):
        """Test that an error is raised for an empty distribution."""
        logprobs = np.array([])
        with self.assertRaises(ValueError):
            entropy_from_logprobs(logprobs)

    @given(st.lists(st.floats(min_value=-10, max_value=0), min_size=1, max_size=10).map(np.array))
    def test_entropy_from_logprobs_property(self, logprobs):
        """Test entropy calculation with property-based testing."""
        # Calculate entropy
        entropy = entropy_from_logprobs(logprobs)

        # Entropy should be non-negative
        self.assertGreaterEqual(entropy, 0.0)

        # For uniform distribution, entropy should be close to log(n)
        n = len(logprobs)
        if n > 0 and np.allclose(logprobs, logprobs[0]):
            # For uniform distribution with probabilities p = 1/n,
            # entropy = -∑(p * log(p)) = -n * (1/n * log(1/n)) = -log(1/n) = log(n)
            # But in our case, we have log probabilities, so we need to adjust
            uniform_logprob = logprobs[0]
            prob = np.exp(uniform_logprob)
            expected_entropy = -n * (prob * uniform_logprob)
            self.assertAlmostEqual(entropy, expected_entropy, places=4)

    def test_token_entropy_empty(self):
        """Test token entropy calculation for an empty sequence."""
        token_logprobs = []
        entropies = token_entropy(token_logprobs)
        self.assertEmpty(entropies)

    def test_token_entropy_single(self):
        """Test token entropy calculation for a single token."""
        token_logprobs = [np.array([-0.693, -0.693])]  # log(0.5), log(0.5)
        entropies = token_entropy(token_logprobs)
        self.assertLen(entropies, 1)
        self.assertAlmostEqual(entropies[0], 0.693, places=3)

    def test_token_entropy_multiple(self):
        """Test token entropy calculation for multiple tokens."""
        token_logprobs = [
            np.array([-0.693, -0.693]),  # Uniform: log(0.5), log(0.5)
            np.array([-0.223, -1.609]),  # Non-uniform: log(0.8), log(0.2)
        ]
        entropies = token_entropy(token_logprobs)
        self.assertLen(entropies, 2)
        self.assertAlmostEqual(entropies[0], 0.693, places=3)
        self.assertAlmostEqual(entropies[1], 0.5, places=1)

    @given(
        st.lists(
            st.lists(st.floats(min_value=-10, max_value=0), min_size=1, max_size=5).map(np.array),
            min_size=0,
            max_size=5,
        )
    )
    def test_token_entropy_property(self, token_logprobs):
        """Test token entropy calculation with property-based testing."""
        entropies = token_entropy(token_logprobs)

        # Should return a list of the same length
        self.assertLen(entropies, len(token_logprobs))

        # Each entropy should be non-negative
        for entropy in entropies:
            self.assertGreaterEqual(entropy, 0.0)

    def test_mean_entropy_empty(self):
        """Test that an error is raised for an empty sequence."""
        token_logprobs = []
        with self.assertRaises(ValueError):
            mean_entropy(token_logprobs)

    def test_mean_entropy_single(self):
        """Test mean entropy calculation for a single token."""
        token_logprobs = [np.array([-0.693, -0.693])]  # log(0.5), log(0.5)
        entropy = mean_entropy(token_logprobs)
        self.assertAlmostEqual(entropy, 0.693, places=3)

    def test_mean_entropy_multiple(self):
        """Test mean entropy calculation for multiple tokens."""
        token_logprobs = [
            np.array([-0.693, -0.693]),  # Uniform: log(0.5), log(0.5)
            np.array([-0.223, -1.609]),  # Non-uniform: log(0.8), log(0.2)
        ]
        entropy = mean_entropy(token_logprobs)
        # Mean of 0.693 and 0.5
        self.assertAlmostEqual(entropy, 0.597, places=3)

    def test_mean_entropy_with_weights(self):
        """Test mean entropy calculation with weights."""
        token_logprobs = [
            np.array([-0.693, -0.693]),  # Uniform: log(0.5), log(0.5)
            np.array([-0.223, -1.609]),  # Non-uniform: log(0.8), log(0.2)
        ]
        weights = [0.2, 0.8]
        entropy = mean_entropy(token_logprobs, weights)
        # Weighted mean: 0.2*0.693 + 0.8*0.5 = 0.1386 + 0.4 = 0.5386
        self.assertAlmostEqual(entropy, 0.539, places=3)

    def test_mean_entropy_weights_mismatch(self):
        """Test that an error is raised if weights length doesn't match."""
        token_logprobs = [
            np.array([-0.693, -0.693]),
            np.array([-0.223, -1.609]),
        ]
        weights = [0.5]
        with self.assertRaises(ValueError):
            mean_entropy(token_logprobs, weights)

    @given(
        st.lists(
            st.lists(st.floats(min_value=-10, max_value=0), min_size=1, max_size=5).map(np.array),
            min_size=1,
            max_size=5,
        )
    )
    def test_mean_entropy_property(self, token_logprobs):
        """Test mean entropy calculation with property-based testing."""
        # Calculate individual entropies
        entropies = token_entropy(token_logprobs)

        # Calculate mean entropy
        mean = mean_entropy(token_logprobs)

        # Mean entropy should equal mean of individual entropies
        expected_mean = sum(entropies) / len(entropies)
        self.assertAlmostEqual(mean, expected_mean, places=5)

        # Mean entropy should be bounded by min and max entropies
        if entropies:
            self.assertLessEqual(mean, max(entropies))
            self.assertGreaterEqual(mean, min(entropies))


if __name__ == "__main__":
    absltest.main()
