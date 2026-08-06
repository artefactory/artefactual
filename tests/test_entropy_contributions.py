import numpy as np
import pytest
from conftest import rank_vectors
from hypothesis import given

from artefactual.scoring import EntropyContributionsMixin

entropy_contributions = EntropyContributionsMixin.entropy_contributions


def test_compute_entropy_contributions_basic():
    """Contributions are in nats, as the ECIR2026 release computes them."""
    # p = 0.5 twice: each contributes -0.5*ln(0.5), summing to ln 2 -- a fair coin's
    # entropy in nats, which is the value that pins the log base.
    logprobs = np.array([[-0.69314718, -0.69314718]], dtype=np.float32)

    result = EntropyContributionsMixin.entropy_contributions(logprobs)

    expected_entropy = -0.5 * np.log(0.5)  # ln 2 nats in total
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

    np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-12)


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


# --- invariants of s_kj = -p*log(p) ----------------------------------------------------


@given(ranks=rank_vectors())
def test_contributions_are_non_negative(ranks):
    # p in (0, 1] and log(p) <= 0, so -p*log(p) >= 0; a negative value would flip the
    # sign of every downstream weighted sum
    result = entropy_contributions(np.array([ranks]))
    assert (result >= 0).all()


@given(ranks=rank_vectors())
def test_contributions_preserve_shape(ranks):
    array = np.array([ranks])
    assert entropy_contributions(array).shape == array.shape


@given(ranks=rank_vectors(min_ranks=2))
def test_contributions_are_invariant_to_input_order(ranks):
    # the function re-sorts internally, so callers need not pre-sort
    shuffled = list(reversed(ranks))
    np.testing.assert_allclose(
        entropy_contributions(np.array([ranks])),
        entropy_contributions(np.array([shuffled])),
        rtol=1e-6,
    )


def test_contributions_are_not_monotonic_in_rank():
    """-p*log(p) peaks at p = 1/e, so rank order is not contribution order.

    `LogProbParser` truncates by rank, which therefore does *not* drop the smallest
    contributions. Pinning this stops anyone from "optimising" the truncation into a
    top-k-by-magnitude, which would change the metric.
    """
    # rank 1 is nearly certain (tiny contribution), rank 2 sits at the p = 1/e peak
    result = entropy_contributions(np.array([[-0.01, -1.0]]))[0]

    assert result[0] < result[1]


def test_the_peak_contribution_is_at_logprob_minus_one():
    grid = np.linspace(-6.0, 0.0, 601)
    contributions = entropy_contributions(grid.reshape(1, -1))[0]
    # the array comes back sorted descending by logprob, so recover the argmax position
    peak = np.sort(grid)[::-1][int(np.argmax(contributions))]

    assert peak == pytest.approx(-1.0, abs=0.05)


def test_a_certain_token_contributes_nothing():
    np.testing.assert_allclose(entropy_contributions(np.array([[0.0]]))[0, 0], 0.0, atol=1e-12)


def test_padded_positions_stay_nan():
    # LogProbParser pads short sequences with NaN and the reductions rely on it surviving
    result = entropy_contributions(np.array([[-0.5, np.nan]]))
    assert np.isnan(result).sum() == 1
