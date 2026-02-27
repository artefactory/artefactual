"""
Property-based behavioral tests for SentenceProbabilityScorer.

These tests verify the long-term correctness and numerical robustness of the
scorer independently of the existing example-based tests in test_sentence_proba.py.

Contract being tested:
  - compute(inputs) returns one probability per sequence, in (0, 1].
  - compute_token_scores(inputs) returns one array of token probabilities per
    sequence; each token probability is in (0, 1].
  - compute and compute_token_scores are consistent:
      compute(inputs)[i] ≈ prod(compute_token_scores(inputs)[i])
  - The scorer is monotone: sequences with higher log probabilities produce
    higher sentence probabilities.
  - Adding tokens with logprob < 0 to a sequence strictly decreases its
    sentence probability.
  - The scorer is deterministic.
  - The scorer accepts the list-of-arrays format returned by the parsers
    (production path for variable-length sequences).
  - Empty sequences must not silently produce probability 1.0
    (exp(sum([])) = exp(0) = 1.0 is the maximum confidence score — misleading
    when no logprob data was actually returned by the model).
  - Numerical underflow (very long / low-confidence sequences) must be
    surfaced as a warning or error rather than silently producing 0.0.
"""

import warnings

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from artefactual.scoring.probability_methods.sentence_proba import SentenceProbabilityScorer

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

logprob_st = st.floats(min_value=-20.0, max_value=-1e-6, allow_nan=False, allow_infinity=False)
nonempty_seq_st = st.lists(logprob_st, min_size=1, max_size=50)
sequences_st = st.lists(nonempty_seq_st, min_size=1, max_size=5)


def _as_list_of_arrays(lol: list[list[float]]) -> list[np.ndarray]:
    """Convert list-of-lists to the list-of-arrays format returned by parsers."""
    return [np.array(seq, dtype=np.float64) for seq in lol]


# ---------------------------------------------------------------------------
# Value range: outputs are valid probabilities
# ---------------------------------------------------------------------------


@given(sequences_st)
def test_compute_returns_values_in_zero_one(logprob_lists):
    """
    Sentence probabilities must lie in (0, 1].

    Computed as exp(sum of logprobs).  Since every logprob is ≤ 0,
    the sum is ≤ 0 and exp(≤0) ∈ (0, 1].
    """
    scorer = SentenceProbabilityScorer()
    result = scorer.compute(_as_list_of_arrays(logprob_lists))
    assert all(0.0 < p <= 1.0 for p in result), (
        f"Expected all probabilities in (0, 1], got: {result}"
    )


@given(sequences_st)
def test_compute_token_scores_all_in_zero_one(logprob_lists):
    """Token-level probabilities must lie in (0, 1]."""
    scorer = SentenceProbabilityScorer()
    result = scorer.compute_token_scores(_as_list_of_arrays(logprob_lists))
    for seq in result:
        assert np.all((seq > 0.0) & (seq <= 1.0)), (
            f"Expected token probabilities in (0, 1], got: {seq}"
        )


@given(sequences_st)
def test_compute_returns_finite_values(logprob_lists):
    """All sentence probabilities must be finite (no NaN, no ±inf)."""
    scorer = SentenceProbabilityScorer()
    result = scorer.compute(_as_list_of_arrays(logprob_lists))
    assert all(np.isfinite(p) for p in result)


# ---------------------------------------------------------------------------
# Consistency between compute and compute_token_scores
# ---------------------------------------------------------------------------


@given(sequences_st)
def test_compute_equals_product_of_token_scores(logprob_lists):
    """
    compute(inputs)[i] must equal the product of compute_token_scores(inputs)[i].

    The sentence probability is the joint probability of all tokens, so these
    two methods must stay consistent as the implementation evolves.
    """
    scorer = SentenceProbabilityScorer()
    inputs = _as_list_of_arrays(logprob_lists)
    seq_scores = scorer.compute(inputs)
    tok_scores = scorer.compute_token_scores(inputs)

    for seq_prob, tok_probs in zip(seq_scores, tok_scores):
        expected = float(np.prod(tok_probs))
        assert seq_prob == pytest.approx(expected, rel=1e-5), (
            f"compute={seq_prob} does not match product of token scores={expected}"
        )


# ---------------------------------------------------------------------------
# Cardinality and shape invariants
# ---------------------------------------------------------------------------


@given(sequences_st)
def test_compute_returns_one_score_per_sequence(logprob_lists):
    """len(result) must equal the number of input sequences."""
    scorer = SentenceProbabilityScorer()
    result = scorer.compute(_as_list_of_arrays(logprob_lists))
    assert len(result) == len(logprob_lists)


@given(sequences_st)
def test_compute_token_scores_length_matches_input(logprob_lists):
    """Each token-score array must have the same length as the input sequence."""
    scorer = SentenceProbabilityScorer()
    result = scorer.compute_token_scores(_as_list_of_arrays(logprob_lists))
    for tok_scores, expected in zip(result, logprob_lists):
        assert len(tok_scores) == len(expected)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


@given(
    st.integers(min_value=1, max_value=20),
    nonempty_seq_st,
    st.lists(
        st.floats(min_value=1e-4, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    ),
)
def test_higher_logprob_sequence_has_higher_sentence_probability(n, lps_base, deltas):
    """
    A sequence whose every token logprob is strictly higher than the
    corresponding token in another sequence must produce a strictly higher
    sentence probability.

    Generated directly by adding a positive delta to each element, avoiding
    the heavy filtering that would result from assume() on independently sampled lists.
    """
    n = min(n, len(lps_base), len(deltas))
    lps_low = lps_base[:n]
    lps_high = [min(lp + d, -1e-9) for lp, d in zip(lps_low, deltas[:n])]
    assume(all(h > l for h, l in zip(lps_high, lps_low)))

    scorer = SentenceProbabilityScorer()
    result_high, result_low = scorer.compute(
        [np.array(lps_high), np.array(lps_low)]
    )
    assert result_high > result_low


@given(nonempty_seq_st, logprob_st)
def test_appending_token_decreases_sentence_probability(lps, extra_lp):
    """
    Appending a token with logprob < 0 must strictly decrease the sentence
    probability: exp(sum + extra_lp) < exp(sum) when extra_lp < 0.
    """
    scorer = SentenceProbabilityScorer()
    short_score = scorer.compute([np.array(lps)])[0]
    long_score = scorer.compute([np.array(lps + [extra_lp])])[0]
    assert long_score < short_score


# ---------------------------------------------------------------------------
# List-of-arrays input (production path from parsers)
# ---------------------------------------------------------------------------


@given(sequences_st)
def test_list_of_arrays_input_works(logprob_lists):
    """
    The production path from vllm_sampled_tokens_logprobs and the OpenAI
    parsers returns a list of 1-D numpy arrays when sequences differ in length.
    SentenceProbabilityScorer.compute must accept this format.
    """
    scorer = SentenceProbabilityScorer()
    inputs = _as_list_of_arrays(logprob_lists)
    result = scorer.compute(inputs)
    assert len(result) == len(logprob_lists)
    assert all(0.0 < p <= 1.0 for p in result)


# ---------------------------------------------------------------------------
# Empty sequences — must not silently return 1.0
# ---------------------------------------------------------------------------


def test_empty_sequence_does_not_silently_return_one():
    """
    An empty sequence produces np.sum([]) = 0.0 → exp(0) = 1.0, the maximum
    confidence score.  For an uncertainty library this is the worst possible
    silent failure: a sequence with no logprob data appears maximally confident.

    The scorer must raise a ValueError rather than silently returning 1.0.
    """
    scorer = SentenceProbabilityScorer()
    with pytest.raises(ValueError):
        scorer.compute([np.array([])])


@given(st.integers(min_value=1, max_value=5))
def test_multiple_empty_sequences_raise(n_empty):
    """Multiple empty sequences must all be rejected, not produce a list of 1.0s."""
    scorer = SentenceProbabilityScorer()
    with pytest.raises(ValueError):
        scorer.compute([np.array([]) for _ in range(n_empty)])


# ---------------------------------------------------------------------------
# Numerical underflow — must surface as warning, not silent 0.0
# ---------------------------------------------------------------------------


@given(st.integers(min_value=1500, max_value=3000), logprob_st)
@settings(max_examples=20)
def test_very_long_sequence_underflow_is_surfaced(n_tokens, lp_per_token):
    """
    Sequences long enough to push the log-probability sum below the float64
    underflow boundary (~-745) produce exp(sum) = 0.0 silently.

    This assigns the lowest possible confidence score to an arbitrarily long
    valid response with no indication to the caller.  The scorer must emit a
    RuntimeWarning (or raise) so callers can detect the issue and switch to
    log-space arithmetic.

    float64 underflow boundary: exp(-745) ≈ 5e-324 ≈ 0.0
    """
    log_sum = n_tokens * lp_per_token
    assume(log_sum < -745.0)

    scorer = SentenceProbabilityScorer()
    seq = [np.array([lp_per_token] * n_tokens, dtype=np.float64)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = scorer.compute(seq)

    underflowed = result[0] == 0.0
    warned = any(issubclass(w.category, (RuntimeWarning, UserWarning)) for w in caught)

    assert not underflowed or warned, (
        f"Underflow to 0.0 (log_sum={log_sum:.1f}) with no warning. "
        "The caller has no way to detect the invalid score."
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@given(sequences_st)
def test_compute_is_deterministic(logprob_lists):
    """Calling compute twice with the same input must produce identical output."""
    scorer = SentenceProbabilityScorer()
    inputs = _as_list_of_arrays(logprob_lists)
    assert scorer.compute(inputs) == scorer.compute(inputs)


@given(sequences_st)
def test_compute_token_scores_is_deterministic(logprob_lists):
    """Calling compute_token_scores twice with the same input must produce identical output."""
    scorer = SentenceProbabilityScorer()
    inputs = _as_list_of_arrays(logprob_lists)
    r1 = scorer.compute_token_scores(inputs)
    r2 = scorer.compute_token_scores(inputs)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a, b)
