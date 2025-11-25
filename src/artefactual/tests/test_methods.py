"""Tests for the scoring.methods module."""

from itertools import chain, repeat

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from absl.testing import absltest, parameterized
from artefactual.scoring.methods import ScoringMethod, score_fn
from hypothesis import given

N_TEST_SAMPLES = 200


class ScoreFnTest(parameterized.TestCase):
    """Test cases for the score_fn function."""

    def test_naive_scoring(self):
        """Test the naive scoring method."""
        logprobs = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
        scores = score_fn(ScoringMethod.NAIVE, logprobs)
        # exp(-1) * exp(-2) * exp(-3) = exp(-6) ≈ 0.0025
        self.assertAlmostEqual(scores[0], 0.0025, places=4)

    def test_naive_scoring_batch(self):
        """Test the naive scoring method with a batch of inputs."""
        logprobs = np.array(
            [
                [-1.0, -2.0, -3.0],
                [-0.5, -1.5, -2.5],
            ],
            dtype=np.float32,
        )
        scores = score_fn(ScoringMethod.NAIVE, logprobs)
        # First sample: exp(-1) * exp(-2) * exp(-3) = exp(-6) ≈ 0.0025
        # Second sample: exp(-0.5) * exp(-1.5) * exp(-2.5) = exp(-4.5) ≈ 0.0111
        self.assertAlmostEqual(scores[0], 0.0025, places=4)
        self.assertAlmostEqual(scores[1], 0.0111, places=4)

    def test_mean_scoring(self):
        """Test the mean scoring method."""
        logprobs = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
        scores = score_fn(ScoringMethod.MEAN, logprobs)
        # exp((-1 - 2 - 3) / 3) = exp(-2) ≈ 0.1353
        self.assertAlmostEqual(scores[0], 0.1353, places=4)

    def test_mean_scoring_batch(self):
        """Test the mean scoring method with a batch of inputs."""
        logprobs = np.array(
            [
                [-1.0, -2.0, -3.0],
                [-0.5, -1.5, -2.5],
            ],
            dtype=np.float32,
        )
        scores = score_fn(ScoringMethod.MEAN, logprobs)
        # First sample: exp((-1 - 2 - 3) / 3) = exp(-2) ≈ 0.1353
        # Second sample: exp((-0.5 - 1.5 - 2.5) / 3) = exp(-1.5) ≈ 0.2231
        self.assertAlmostEqual(scores[0], 0.1353, places=4)
        self.assertAlmostEqual(scores[1], 0.2231, places=4)

    @given(
        logprobs=hnp.arrays(
            dtype=np.float32,
            shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=N_TEST_SAMPLES, max_side=N_TEST_SAMPLES),
            elements=st.floats(-5, 0, allow_subnormal=False, width=32),
        ),
    )
    def test_supervised_sigmoid(self, logprobs):
        """Test the supervised sigmoid scoring method."""
        ids = np.arange(N_TEST_SAMPLES)
        labels = np.asarray(list(chain.from_iterable([repeat(0, N_TEST_SAMPLES // 2), repeat(1, N_TEST_SAMPLES // 2)])))
        test_ids, scores, test_labels = score_fn(ScoringMethod.SUPERVISED_SIGMOID, logprobs, ids, labels)

        # Check dimensions
        self.assertEqual(len(test_ids), len(scores))
        self.assertEqual(len(test_ids), len(test_labels))

        # Check score ranges
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    @given(
        logprobs=hnp.arrays(
            dtype=np.float32,
            shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=N_TEST_SAMPLES, max_side=N_TEST_SAMPLES),
            elements=st.floats(-5, 0, allow_subnormal=False, width=32),
        ),
    )
    def test_supervised_isotonic(self, logprobs):
        ids = np.arange(N_TEST_SAMPLES)
        labels = np.asarray(list(chain.from_iterable([repeat(0, N_TEST_SAMPLES // 2), repeat(1, N_TEST_SAMPLES // 2)])))
        test_ids, scores, test_labels = score_fn(ScoringMethod.SUPERVISED_ISOTONIC, logprobs, ids, labels)

        # Check dimensions
        self.assertEqual(len(test_ids), len(scores))
        self.assertEqual(len(test_ids), len(test_labels))

        # Check score ranges
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_supervised_length_mismatch(self):
        """Test the supervised scoring method with mismatched lengths."""
        logprobs = np.array([[-1.0, -2.0, -3.0]], dtype=np.float32)
        ids = np.array([1, 2])
        labels = np.array([1])

        with self.assertRaises(ValueError):
            score_fn(ScoringMethod.SUPERVISED_SIGMOID, logprobs, ids, labels)

    def test_naive_scoring_property(self):
        """Test the naive scoring method with a range of inputs."""
        # Create a range of test cases with different dimensions
        test_cases = [
            np.array([[-1.0, -2.0, -3.0]], dtype=np.float32),
            np.array([[-0.5, -1.5, -2.5], [-3.0, -4.0, -5.0]], dtype=np.float32),
            np.array([[-0.1], [-0.2], [-0.3]], dtype=np.float32),
        ]

        for logprobs in test_cases:
            # Calculate expected scores
            expected_scores = np.exp(np.sum(logprobs, axis=1))

            # Get actual scores
            scores = score_fn(ScoringMethod.NAIVE, logprobs)

            # Compare
            np.testing.assert_allclose(scores, expected_scores, rtol=1e-5)

    def test_mean_scoring_property(self):
        """Test the mean scoring method with a range of inputs."""
        # Create a range of test cases with different dimensions
        test_cases = [
            np.array([[-1.0, -2.0, -3.0]], dtype=np.float32),
            np.array([[-0.5, -1.5, -2.5], [-3.0, -4.0, -5.0]], dtype=np.float32),
            np.array([[-0.1], [-0.2], [-0.3]], dtype=np.float32),
        ]

        for logprobs in test_cases:
            # Calculate expected scores
            expected_scores = np.exp(np.mean(logprobs, axis=1))

            # Get actual scores
            scores = score_fn(ScoringMethod.MEAN, logprobs)

            # Compare
            np.testing.assert_allclose(scores, expected_scores, rtol=1e-5)


if __name__ == "__main__":
    absltest.main()
