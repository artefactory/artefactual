"""Numerical correctness of the EPR and WEPR feature extraction.

The features are the whole method: a coefficient vector is meaningless if the quantity it
multiplies is not the one the weights were trained on, and an error there is invisible
downstream — every reduction returns a plausible float, and a constant scale error still
ranks correctly, so it survives every shape and ordering check in the rest of the suite.

The oracle is `ecir2026_reference`, a frozen copy of the arithmetic from the tagged release
that produced the paper's results. It is the authority rather than the paper's equations,
because the published coefficients were trained against it, and the two differ: Eq. 1 and 3
are written in bits, while the release computes in nats — a factor of ln 2 that a fitted
coefficient absorbs.

Where the release and the paper agree, both are pinned. EPR sums the rank axis (Eq. 3) and
averages over tokens (Eq. 6); WEPR's 2K features reproduce Eq. 8 when dotted with
`[beta, gamma]`. Where they differ, the release wins, and the analytic anchors below are
written in nats to say so unambiguously.

`EntropyTransformer` also treats a fully-NaN token as padding and excludes it, which the
release had no notion of — it scored one sequence at a time. That is called out where it applies below.
"""

import math

import numpy as np
import pytest
from conftest import logprob_cubes
from ecir2026_reference import compute_entropy_contributions, sequence_epr, sequence_wepr
from hypothesis import given
from hypothesis import strategies as st

from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer

# --- EPR against the release -----------------------------------------------------------


@given(logprobs=logprob_cubes())
def test_epr_matches_the_ecir2026_release(logprobs):
    produced = EntropyTransformer(reduction="epr").transform(logprobs)
    k = logprobs.shape[-1]

    for index, sequence in enumerate(logprobs):
        assert produced[index, 0] == pytest.approx(sequence_epr(sequence, k), rel=1e-5, abs=1e-12)


@given(logprobs=logprob_cubes())
def test_the_contributions_match_the_release_rank_for_rank(logprobs):
    # the per-rank table both reductions are built from, before any pooling
    k = logprobs.shape[-1]

    produced = EntropyTransformer().entropy_contributions(logprobs)

    for index, sequence in enumerate(logprobs):
        expected = compute_entropy_contributions(np.sort(sequence, axis=-1)[:, ::-1], k)
        np.testing.assert_allclose(produced[index], expected, rtol=1e-5, atol=1e-12)


# --- EPR analytic anchors, in nats as the release computes them -------------------------


@pytest.mark.parametrize("k", [2, 3, 8, 15, 20])
def test_a_uniform_distribution_gives_exactly_ln_k(k):
    """The anchor that pins the reduction and the log base together.

    A uniform distribution over k outcomes has entropy ln(k) nats. Averaging the rank axis
    instead of summing would give ln(k)/k; computing in bits would give log2(k). Only the
    release's formulation lands on ln(k).
    """
    uniform = np.full((1, 1, k), np.log(1 / k))

    epr = EntropyTransformer(reduction="epr").transform(uniform)

    assert epr[0, 0] == pytest.approx(math.log(k), rel=1e-6)


def test_a_certain_token_has_zero_entropy():
    # p = 1 for the top rank, everything else vanishing: -1*ln(1) = 0
    certain = np.array([[[0.0, -50.0, -50.0]]])

    assert EntropyTransformer(reduction="epr").transform(certain)[0, 0] == pytest.approx(0.0, abs=1e-8)


def test_a_fair_coin_matches_hand_arithmetic():
    # p = (0.5, 0.5): each rank contributes 0.5*ln2, so H = ln 2 nats
    coin = np.array([[[math.log(0.5), math.log(0.5)]]])

    assert EntropyTransformer(reduction="epr").transform(coin)[0, 0] == pytest.approx(math.log(2), rel=1e-6)


def test_epr_is_a_rate_not_a_total():
    """Repeating a token must not change EPR — it is a mean over the sequence, per Eq. 6."""
    one = np.array([[[math.log(0.5), math.log(0.5)]]])
    three = np.repeat(one, 3, axis=1)

    transformer = EntropyTransformer(reduction="epr")

    assert transformer.transform(three)[0, 0] == pytest.approx(transformer.transform(one)[0, 0], rel=1e-6)


@given(logprobs=logprob_cubes())
def test_epr_is_bounded_by_its_rank_count_over_e(logprobs):
    """The bound is K/e, not ln(K).

    ln(K) bounds the entropy of a *normalised* distribution, and the top-K probabilities
    are not one — they are a truncated tail of the vocabulary and sum to less than 1. Each
    contribution `-p ln p` peaks at `p = 1/e` where it equals `1/e`, so K of them cannot
    exceed `K/e`, and that bound is reachable: K ranks all at `p = 1/e` attain it.
    """
    k = logprobs.shape[-1]

    epr = EntropyTransformer(reduction="epr").transform(logprobs)

    assert (epr[:, 0] <= k / math.e + 1e-6).all()
    assert (epr[:, 0] >= -1e-9).all()


def test_the_bound_is_attained_by_ranks_at_the_peak():
    # every rank sitting exactly at p = 1/e, the worst case the bound above allows
    k = 5
    at_peak = np.full((1, 1, k), -1.0)  # ln(1/e) = -1

    epr = EntropyTransformer(reduction="epr").transform(at_peak)

    assert epr[0, 0] == pytest.approx(k / math.e, rel=1e-6)


# --- WEPR against the release ------------------------------------------------------------


@given(logprobs=logprob_cubes(), seed=st.integers(0, 2**16))
def test_the_wepr_features_reproduce_the_release_score(logprobs, seed):
    """The features are correct only if a dot product with (beta, gamma) gives Eq. 8.

    That is exactly what the fitted classifier does, so this pins the contract between the
    transformer and the coefficients, not the transformer alone.
    """
    rng = np.random.default_rng(seed)
    k = logprobs.shape[-1]
    mean_weights, max_weights = rng.normal(size=k), rng.normal(size=k)
    intercept = float(rng.normal())

    features = EntropyTransformer(reduction="wepr").transform(logprobs)

    for index, sequence in enumerate(logprobs):
        expected = sequence_wepr(sequence, k, mean_weights, max_weights, intercept)
        produced = float(features[index] @ np.concatenate([mean_weights, max_weights])) + intercept

        assert produced == pytest.approx(expected, rel=1e-5, abs=1e-12)


@given(logprobs=logprob_cubes())
def test_the_wepr_branches_are_the_per_rank_mean_and_max(logprobs):
    k = logprobs.shape[-1]
    features = EntropyTransformer(reduction="wepr").transform(logprobs)

    for index, sequence in enumerate(logprobs):
        s_kj = compute_entropy_contributions(np.sort(sequence, axis=-1)[:, ::-1], k)
        np.testing.assert_allclose(features[index, :k], s_kj.mean(axis=0), rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(features[index, k:], s_kj.max(axis=0), rtol=1e-5, atol=1e-12)


@given(logprobs=logprob_cubes())
def test_the_wepr_mean_branch_never_exceeds_the_max_branch(logprobs):
    # a per-rank mean over tokens cannot exceed that rank's maximum
    features = EntropyTransformer(reduction="wepr").transform(logprobs)
    k = logprobs.shape[-1]

    assert (features[:, :k] <= features[:, k:] + 1e-9).all()


def test_wepr_reduces_to_epr_when_every_mean_coefficient_is_one():
    """Summing the mean branch is the mean over tokens of the sum over ranks — EPR itself.

    The two reductions must agree there, or one of them is reading a different quantity.
    """
    logprobs = np.array([[[math.log(0.5), math.log(0.3), math.log(0.2)]] * 4])

    epr = EntropyTransformer(reduction="epr").transform(logprobs)[0, 0]
    wepr = EntropyTransformer(reduction="wepr").transform(logprobs)[0]

    assert wepr[:3].sum() == pytest.approx(epr, rel=1e-6)


# --- padding, which the release had no notion of -----------------------------------------


def test_padded_tokens_are_excluded_from_both_reductions():
    """A padded token is absent, not zero-entropy: counting it would drag EPR down.

    The release scored one sequence at a time and never padded, so this behaviour is this
    implementation's own — it is what makes a batched score equal an unbatched one.
    """
    real = np.array([[[math.log(0.5), math.log(0.5)]]])
    padded = np.array([[[math.log(0.5), math.log(0.5)], [np.nan, np.nan]]])

    for reduction in ("epr", "wepr"):
        transformer = EntropyTransformer(reduction=reduction)
        np.testing.assert_allclose(transformer.transform(padded), transformer.transform(real), rtol=1e-6)
