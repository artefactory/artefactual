"""Tests for the scoring.logprobs module."""

from dataclasses import dataclass

import numpy as np
from absl.testing import absltest, parameterized
from artefactual.scoring.logprobs import (
    extract_logprobs,
    process_logprobs,
)
from hypothesis import given
from hypothesis import strategies as st


@dataclass
class TestLogProbValue:
    """Test implementation of LogProbValue."""

    logprob: float


class LogprobsTest(parameterized.TestCase):
    """Test cases for the logprobs module."""

    def test_process_logprobs_empty(self):
        """Test processing empty logprobs."""
        logprobs = []
        result = process_logprobs(logprobs, max_len=3)
        self.assertEqual(result.shape, (0, 3))

    def test_process_logprobs_single(self):
        """Test processing a single sequence of logprobs."""
        logprobs = [[0.1, 0.2, 0.3]]
        result = process_logprobs(logprobs, max_len=3)
        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_almost_equal(result[0], [0.1, 0.2, 0.3])

    def test_process_logprobs_multiple(self):
        """Test processing multiple sequences of logprobs."""
        logprobs = [[0.1, 0.2, 0.3], [0.4, 0.5]]
        result = process_logprobs(logprobs, max_len=3)
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_almost_equal(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_almost_equal(result[1], [0.4, 0.5, 0.0])

    def test_process_logprobs_truncate(self):
        """Test that logprobs are truncated if longer than max_len."""
        logprobs = [[0.1, 0.2, 0.3, 0.4]]
        result = process_logprobs(logprobs, max_len=3)
        self.assertEqual(result.shape, (1, 3))
        np.testing.assert_almost_equal(result[0], [0.1, 0.2, 0.3])

    def test_process_logprobs_invalid_max_len(self):
        """Test that an error is raised if max_len is invalid."""
        logprobs = [[0.1, 0.2, 0.3]]
        with self.assertRaises(ValueError):
            process_logprobs(logprobs, max_len=0)
        with self.assertRaises(ValueError):
            process_logprobs(logprobs, max_len=-1)

    @given(
        st.lists(st.lists(st.floats(min_value=-10, max_value=0), max_size=5), max_size=3),
        st.integers(min_value=1, max_value=10),
    )
    def test_process_logprobs_property(self, logprobs, max_len):
        """Test processing logprobs with property-based testing."""
        result = process_logprobs(logprobs, max_len=max_len)
        self.assertEqual(result.shape, (len(logprobs), max_len))
        for i, lp in enumerate(logprobs):
            truncated = lp[:max_len]
            # Convert truncated to float32 for accurate comparison with the result
            truncated_float32 = np.array(truncated, dtype=np.float32)
            np.testing.assert_array_equal(result[i, : len(truncated)], truncated_float32)
            np.testing.assert_array_equal(
                result[i, len(truncated) :], np.zeros(max_len - len(truncated), dtype=np.float32)
            )

    def test_extract_logprobs_empty(self):
        """Test extracting from empty logprobs."""
        logprobs = []
        tokens, lps = extract_logprobs(logprobs)
        self.assertEmpty(tokens)
        self.assertEmpty(lps)

    def test_extract_logprobs_single(self):
        """Test extracting from a single dictionary of logprobs."""
        logprobs = [{1: TestLogProbValue(-0.5), 2: TestLogProbValue(-1.2)}]
        tokens, lps = extract_logprobs(logprobs)
        self.assertSequenceEqual(tokens, (1, 2))
        self.assertSequenceEqual(lps, (-0.5, -1.2))

    def test_extract_logprobs_multiple(self):
        """Test extracting from multiple dictionaries of logprobs."""
        logprobs = [
            {1: TestLogProbValue(-0.5), 2: TestLogProbValue(-1.2)},
            {3: TestLogProbValue(-0.8)},
        ]
        tokens, lps = extract_logprobs(logprobs)
        self.assertSequenceEqual(tokens, (1, 2, 3))
        self.assertSequenceEqual(lps, (-0.5, -1.2, -0.8))

    @given(
        st.lists(
            st.dictionaries(
                st.integers(min_value=1, max_value=100),
                st.builds(TestLogProbValue, st.floats(min_value=-10, max_value=0)),
                max_size=5,
            ),
            max_size=3,
        )
    )
    def test_extract_logprobs_property(self, logprobs):
        """Test extracting logprobs with property-based testing."""
        tokens, lps = extract_logprobs(logprobs)

        # Reconstruct flattened tokens and logprobs
        expected_tokens = []
        expected_lps = []
        for d in logprobs:
            for token, lp in d.items():
                expected_tokens.append(token)
                expected_lps.append(lp.logprob)

        self.assertSequenceEqual(set(tokens), set(expected_tokens))
        self.assertEqual(len(lps), len(expected_lps))

        # Match corresponding logprobs with tokens
        token_to_lp = dict(zip(tokens, lps, strict=False))
        for d in logprobs:
            for token, lp in d.items():
                if token in token_to_lp:
                    self.assertIn(
                        token_to_lp[token], [lp.logprob for lp in [d[t] for d in logprobs for t in d if t == token]]
                    )


if __name__ == "__main__":
    absltest.main()
