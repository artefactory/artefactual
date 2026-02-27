"""
Behavioral tests for vllm_sampled_tokens_logprobs.

These tests verify the long-term correctness and robustness of the function
that extracts sampled-token log probabilities from vLLM RequestOutput objects.
They use property-based testing (Hypothesis) to cover the space of realistic
inputs rather than a fixed set of hand-crafted examples.

Contract being tested:
  - The function returns one entry per sequence (one per iteration).
  - Each entry contains the log probability of the *sampled* token at each
    position, in order, not the log probability of any top-K alternative.
  - Results are finite negative floats (log probabilities are ≤ 0).
  - The function is deterministic: same input → same output.
  - Sequences with no logprob data produce an empty entry.
  - Variable-length sequences are handled without crashing.
"""

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from artefactual.preprocessing.vllm_parser import vllm_sampled_tokens_logprobs

# ---------------------------------------------------------------------------
# Mock helpers — minimal shims for vLLM data structures
# ---------------------------------------------------------------------------


class MockLogprob:
    """Minimal stand-in for vllm.Logprob."""

    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class MockCompletionOutput:
    """Minimal stand-in for vllm.CompletionOutput."""

    def __init__(self, token_ids: tuple, logprobs: list | None) -> None:
        self.token_ids = token_ids
        self.logprobs = logprobs  # list[dict[int, Logprob]] | None


class MockRequestOutput:
    """Minimal stand-in for vllm.RequestOutput."""

    def __init__(self, outputs: list) -> None:
        self.outputs = outputs


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Valid log probabilities: strictly negative, finite.
logprob_st = st.floats(min_value=-50.0, max_value=-1e-6, allow_nan=False, allow_infinity=False)

# A single token: (sampled_id, sampled_logprob, alt_id, alt_logprob)
# Sampled IDs live in [1, 500], alternative IDs in [501, 1000] so the ranges
# are disjoint and the test can unambiguously verify which value was returned.
token_entry_st = st.tuples(
    st.integers(min_value=1, max_value=500),    # sampled token id
    logprob_st,                                  # sampled logprob
    st.integers(min_value=501, max_value=1000),  # alternative token id
    logprob_st,                                  # alternative logprob
)

sequence_st = st.lists(token_entry_st, min_size=1, max_size=30)


def _make_comp(tokens: list) -> MockCompletionOutput:
    """Build a MockCompletionOutput from a list of token_entry tuples."""
    token_ids = tuple(t[0] for t in tokens)
    logprobs = [
        {t[0]: MockLogprob(t[1]), t[2]: MockLogprob(t[3])}
        for t in tokens
    ]
    return MockCompletionOutput(token_ids=token_ids, logprobs=logprobs)


def _make_uniform_comp(lps: list[float]) -> MockCompletionOutput:
    """Build a MockCompletionOutput where token id == position index."""
    tids = tuple(range(len(lps)))
    return MockCompletionOutput(
        token_ids=tids,
        logprobs=[{tid: MockLogprob(lp)} for tid, lp in zip(tids, lps)],
    )


# ---------------------------------------------------------------------------
# Correctness: sampled token identity
# ---------------------------------------------------------------------------


@given(sequence_st)
def test_returns_sampled_token_logprob_not_alternative(tokens):
    """
    For every token position the function must return the logprob of the token
    that was actually sampled, not the logprob of any top-K alternative.

    The sampled id lives in [1, 500] and alternatives in [501, 1000], so lookup
    is unambiguous regardless of the relative magnitude of the two logprobs.
    """
    comp = _make_comp(tokens)
    req = MockRequestOutput(outputs=[comp])
    result = vllm_sampled_tokens_logprobs([req], iterations=1)

    expected = [t[1] for t in tokens]
    np.testing.assert_allclose(result[0], expected, rtol=1e-6)


@given(
    st.integers(min_value=1, max_value=20),
    st.lists(logprob_st, min_size=1, max_size=20),
    st.lists(logprob_st, min_size=1, max_size=20),
)
def test_each_iteration_uses_its_own_completion_output(n, lps0_raw, lps1_raw):
    """
    With iterations=2 the function must pair each CompletionOutput with its
    own sequence index, not share logprobs across sequences.

    Regression guard for the variable-shadowing pattern at vllm_parser.py:76
    where the inner comprehension reuses `i` as its loop variable.

    Both sequences are truncated to the same length `n` to isolate this
    concern from the separate ragged-array issue tested below.
    """
    lps0 = (lps0_raw * n)[:n]
    lps1 = (lps1_raw * n)[:n]

    comp0 = _make_uniform_comp(lps0)
    comp1 = _make_uniform_comp(lps1)
    req = MockRequestOutput(outputs=[comp0, comp1])

    result = vllm_sampled_tokens_logprobs([req], iterations=2)

    np.testing.assert_allclose(result[0], lps0, rtol=1e-6)
    np.testing.assert_allclose(result[1], lps1, rtol=1e-6)


# ---------------------------------------------------------------------------
# Correctness: value range
# ---------------------------------------------------------------------------


@given(sequence_st)
def test_all_returned_logprobs_are_non_positive(tokens):
    """
    Log probabilities are logarithms of values in (0, 1], so they must be ≤ 0.
    Any positive value would indicate a bug in the extraction logic.
    """
    comp = _make_comp(tokens)
    result = vllm_sampled_tokens_logprobs([MockRequestOutput([comp])], iterations=1)
    assert np.all(result[0] <= 0.0)


@given(sequence_st)
def test_all_returned_logprobs_are_finite(tokens):
    """Extracted logprobs must be finite (no NaN, no ±inf)."""
    comp = _make_comp(tokens)
    result = vllm_sampled_tokens_logprobs([MockRequestOutput([comp])], iterations=1)
    assert np.all(np.isfinite(result[0]))


# ---------------------------------------------------------------------------
# Shape and cardinality invariants
# ---------------------------------------------------------------------------


@given(sequence_st)
def test_result_length_matches_sequence_length(tokens):
    """One logprob per generated token: len(result[0]) == len(token_ids)."""
    comp = _make_comp(tokens)
    result = vllm_sampled_tokens_logprobs([MockRequestOutput([comp])], iterations=1)
    assert len(result[0]) == len(tokens)


@given(
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=10),
)
def test_number_of_results_matches_iterations(n_seqs, seq_len):
    """len(result) == iterations for any number of equal-length sequences."""
    lp = -0.5
    comp = MockCompletionOutput(
        token_ids=tuple(range(seq_len)),
        logprobs=[{i: MockLogprob(lp)} for i in range(seq_len)],
    )
    req = MockRequestOutput(outputs=[comp] * n_seqs)
    result = vllm_sampled_tokens_logprobs([req], iterations=n_seqs)
    assert len(result) == n_seqs


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


@given(
    st.integers(min_value=1, max_value=10),
    st.lists(logprob_st, min_size=1, max_size=10),
    st.lists(
        st.floats(min_value=1e-6, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
)
def test_higher_logprobs_produce_higher_sentence_score(n, lps_base, deltas):
    """
    A sequence whose every token logprob is strictly higher than the
    corresponding token in another sequence must have a higher total
    log-probability sum.

    Generated directly: lps_high = lps_base + delta (element-wise) so the
    ordering is guaranteed without relying on assume() filtering.
    """
    n = min(n, len(lps_base), len(deltas))
    lps_low = lps_base[:n]
    lps_high = [min(lp + d, -1e-9) for lp, d in zip(lps_low, deltas[:n])]
    assume(all(h > l for h, l in zip(lps_high, lps_low)))  # sanity guard only

    comp_high = _make_uniform_comp(lps_high)
    comp_low = _make_uniform_comp(lps_low)
    req = MockRequestOutput(outputs=[comp_high, comp_low])

    result = vllm_sampled_tokens_logprobs([req], iterations=2)

    assert np.sum(result[0]) > np.sum(result[1])


# ---------------------------------------------------------------------------
# Robustness: variable-length sequences
# ---------------------------------------------------------------------------


@given(
    st.lists(logprob_st, min_size=1, max_size=15),
    st.lists(logprob_st, min_size=1, max_size=15),
)
def test_variable_length_sequences_do_not_crash(lps0, lps1):
    """
    Two sequences of different token counts must be returned without error.

    np.array() on a ragged list raises ValueError in NumPy 2 (vllm_parser.py:79).
    The function must handle variable-length sequences gracefully — returning a
    list of 1-D arrays is the natural solution that avoids the crash and preserves
    per-sequence accuracy.
    """
    assume(len(lps0) != len(lps1))

    req = MockRequestOutput(outputs=[_make_uniform_comp(lps0), _make_uniform_comp(lps1)])
    result = vllm_sampled_tokens_logprobs([req], iterations=2)

    assert len(result) == 2
    np.testing.assert_allclose(result[0], lps0, rtol=1e-6)
    np.testing.assert_allclose(result[1], lps1, rtol=1e-6)


# ---------------------------------------------------------------------------
# Robustness: missing or absent logprob data
# ---------------------------------------------------------------------------


def test_empty_request_output_returns_empty():
    req = MockRequestOutput(outputs=[])
    result = vllm_sampled_tokens_logprobs([req])
    assert len(result) == 0


def test_none_logprobs_produces_empty_entry():
    """A CompletionOutput with logprobs=None yields an empty entry (no crash)."""
    comp = MockCompletionOutput(token_ids=(1, 2), logprobs=None)
    req = MockRequestOutput(outputs=[comp])
    result = vllm_sampled_tokens_logprobs([req], iterations=1)
    assert len(result[0]) == 0


def test_empty_logprobs_list_produces_empty_entry():
    """A CompletionOutput with logprobs=[] yields an empty entry (no crash)."""
    comp = MockCompletionOutput(token_ids=(1,), logprobs=[])
    req = MockRequestOutput(outputs=[comp])
    result = vllm_sampled_tokens_logprobs([req], iterations=1)
    assert len(result[0]) == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@given(sequence_st)
def test_deterministic_output(tokens):
    """Calling the function twice with the same input must produce identical output."""
    comp = _make_comp(tokens)
    req = MockRequestOutput(outputs=[comp])
    r1 = vllm_sampled_tokens_logprobs([req], iterations=1)
    r2 = vllm_sampled_tokens_logprobs([req], iterations=1)
    np.testing.assert_array_equal(r1[0], r2[0])
