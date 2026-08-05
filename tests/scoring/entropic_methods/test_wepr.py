"""Tests for the WEPR scorer.

WEPR reads its own rank count `k` out of the weights file, then aligns every sequence to
that width before the weighted sum. The interesting cases are all about that alignment:
weights and responses are produced by different systems and their `k` need not agree.

Weights are written to real JSON files rather than patched in, so `load_weights` is
exercised on the same path production uses.
"""

import numpy as np
import pytest
from conftest import parsed_sequences, wepr_weights, write_json
from hypothesis import HealthCheck, given, settings

from artefactual.scoring import WEPR


def make_wepr(tmp_path, payload, name="w.json"):
    return WEPR(str(write_json(tmp_path, name, payload)))


def dense_weights(k, *, mean=1.0, maximum=1.0, intercept=0.0):
    coefficients = {}
    for rank in range(1, k + 1):
        coefficients[f"mean_rank_{rank}"] = mean
        coefficients[f"max_rank_{rank}"] = maximum
    return {"intercept": intercept, "coefficients": coefficients}


# --- reading the weights file ----------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3, 15, 32])
def test_k_is_derived_from_the_highest_mean_rank_key(tmp_path, k):
    assert make_wepr(tmp_path, dense_weights(k)).k == k


def test_weights_are_laid_out_in_rank_order(tmp_path):
    payload = {
        "intercept": 0.0,
        "coefficients": {
            "mean_rank_1": 1.0,
            "mean_rank_2": 2.0,
            "mean_rank_3": 3.0,
            "max_rank_1": -1.0,
            "max_rank_2": -2.0,
            "max_rank_3": -3.0,
        },
    }
    scorer = make_wepr(tmp_path, payload)

    np.testing.assert_allclose(scorer.mean_weights, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(scorer.max_weights, [-1.0, -2.0, -3.0])


def test_a_gap_in_the_rank_keys_is_zero_filled(tmp_path):
    # k comes from max(rank), so ranks 2..14 are absent and default to 0.0 rather than
    # raising. Silent, but it keeps the vector width consistent with `k`.
    payload = {"intercept": 0.0, "coefficients": {"mean_rank_1": 1.0, "mean_rank_15": 5.0, "max_rank_1": 1.0}}
    scorer = make_wepr(tmp_path, payload)

    assert scorer.k == 15
    assert scorer.mean_weights[1:14].tolist() == [0.0] * 13
    assert scorer.max_weights[1:].tolist() == [0.0] * 14


def test_weights_without_any_rank_keys_fall_back_to_k_15(tmp_path):
    scorer = make_wepr(tmp_path, {"intercept": 0.5, "coefficients": {}})

    assert scorer.k == 15
    assert not scorer.mean_weights.any()
    assert not scorer.max_weights.any()


def test_non_numeric_rank_suffixes_are_ignored_when_deriving_k(tmp_path):
    payload = {"intercept": 0.0, "coefficients": {"mean_rank_2": 1.0, "mean_rank_total": 9.0}}
    assert make_wepr(tmp_path, payload).k == 2


# --- scoring contract ------------------------------------------------------------------


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(payload=wepr_weights(), batch=parsed_sequences())
def test_compute_returns_one_probability_per_sequence(tmp_path, payload, batch):
    scores = make_wepr(tmp_path, payload).compute(batch)

    assert len(scores) == len(batch)
    assert all(0.0 <= score <= 1.0 for score in scores)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(payload=wepr_weights(), batch=parsed_sequences())
def test_token_scores_keep_one_value_per_token(tmp_path, payload, batch):
    token_scores = make_wepr(tmp_path, payload).compute_token_scores(batch)

    assert len(token_scores) == len(batch)
    for scores, sequence in zip(token_scores, batch, strict=True):
        assert scores.shape == (len(sequence),)
        assert np.all((scores >= 0.0) & (scores <= 1.0))


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(payload=wepr_weights(), batch=parsed_sequences())
def test_scoring_is_independent_of_token_key_insertion_order(tmp_path, payload, batch):
    # sequences arrive as dicts; positions are sorted internally, so a reordered dict
    # must not move the score
    shuffled = [dict(reversed(list(sequence.items()))) for sequence in batch]
    scorer = make_wepr(tmp_path, payload)

    np.testing.assert_allclose(scorer.compute(shuffled), scorer.compute(batch), rtol=1e-6)


def test_empty_batch_scores_nothing(tmp_path):
    scorer = make_wepr(tmp_path, dense_weights(3))
    assert scorer.compute([]) == []
    assert scorer.compute_token_scores([]) == []


@pytest.mark.parametrize("intercept", [-3.0, 0.0, 2.5])
def test_a_token_less_sequence_scores_at_the_baseline(tmp_path, intercept):
    scorer = make_wepr(tmp_path, dense_weights(3, intercept=intercept))

    (score,) = scorer.compute([{}])
    assert score == pytest.approx(1.0 / (1.0 + np.exp(-intercept)))


def test_a_token_less_sequence_has_no_token_scores(tmp_path):
    (scores,) = make_wepr(tmp_path, dense_weights(3)).compute_token_scores([{}])
    assert scores.shape == (0,)


# --- rank-width alignment --------------------------------------------------------------


def test_ranks_beyond_k_are_truncated(tmp_path):
    # weights stop at rank 2, so a 4-rank response must not contribute ranks 3 and 4
    scorer = make_wepr(tmp_path, dense_weights(2))
    wide = [{0: [-0.1, -0.2, -0.3, -0.4]}]
    narrow = [{0: [-0.1, -0.2]}]

    np.testing.assert_allclose(scorer.compute(wide), scorer.compute(narrow), rtol=1e-6)


def test_ranks_below_k_are_zero_padded(tmp_path):
    # -p*log(p) tends to 0 as p tends to 0, so absent ranks must add nothing
    scorer = make_wepr(tmp_path, dense_weights(4))
    narrow = [{0: [-0.1, -0.2]}]

    (score,) = scorer.compute(narrow)
    assert np.isfinite(score)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(batch=parsed_sequences(max_ranks=6))
def test_padding_ranks_with_zero_weight_cannot_change_the_score(tmp_path, batch):
    # widening k while leaving the extra weights at zero is a no-op by construction
    narrow = make_wepr(tmp_path, dense_weights(6), name="narrow.json")

    wide_payload = dense_weights(6)
    for rank in range(7, 13):
        wide_payload["coefficients"][f"mean_rank_{rank}"] = 0.0
        wide_payload["coefficients"][f"max_rank_{rank}"] = 0.0
    wide = make_wepr(tmp_path, wide_payload, name="wide.json")

    np.testing.assert_allclose(wide.compute(batch), narrow.compute(batch), rtol=1e-6)


# --- the weighted sum itself -----------------------------------------------------------


def test_sequence_score_is_the_mean_token_term_plus_the_weighted_max_term(tmp_path):
    # pins Eq. 8 against a hand-computed value so a refactor of the reduction is caught
    scorer = make_wepr(tmp_path, dense_weights(2, mean=1.0, maximum=2.0, intercept=0.5))
    batch = [{0: [-0.5, -1.0], 1: [-0.2, -2.0]}]

    contributions = np.array([
        [-np.exp(-0.5) * -0.5, -np.exp(-1.0) * -1.0],
        [-np.exp(-0.2) * -0.2, -np.exp(-2.0) * -2.0],
    ])
    token_term = (contributions @ np.ones(2)) + 0.5
    max_term = contributions.max(axis=0) @ (2.0 * np.ones(2))
    expected = 1.0 / (1.0 + np.exp(-(token_term.mean() + max_term)))

    assert scorer.compute(batch)[0] == pytest.approx(expected, rel=1e-6)


def test_token_scores_are_the_sigmoid_of_the_weighted_token_sum(tmp_path):
    scorer = make_wepr(tmp_path, dense_weights(2, mean=1.0, intercept=0.5))
    batch = [{0: [-0.5, -1.0]}]

    contributions = np.array([-np.exp(-0.5) * -0.5, -np.exp(-1.0) * -1.0])
    expected = 1.0 / (1.0 + np.exp(-(contributions.sum() + 0.5)))

    np.testing.assert_allclose(scorer.compute_token_scores(batch)[0], [expected], rtol=1e-6)


def test_zero_weights_collapse_every_sequence_to_the_intercept(tmp_path):
    scorer = make_wepr(tmp_path, dense_weights(4, mean=0.0, maximum=0.0, intercept=-1.5))
    batch = [{0: [-0.1, -0.9, -2.0, -3.0]}, {0: [-5.0, -5.0, -5.0, -5.0]}]

    baseline = 1.0 / (1.0 + np.exp(1.5))
    np.testing.assert_allclose(scorer.compute(batch), [baseline, baseline], rtol=1e-6)


def test_a_certain_token_contributes_no_entropy(tmp_path):
    # logprob 0 means p=1, so -p*log(p) is 0 and only the intercept survives
    scorer = make_wepr(tmp_path, dense_weights(1, intercept=0.0))

    (score,) = scorer.compute([{0: [0.0]}])
    assert score == pytest.approx(0.5)


# --- input validation ------------------------------------------------------------------


def test_compute_rejects_a_bare_sequence(tmp_path):
    # beartype guards the batch shape; a single dict is the easy caller mistake
    scorer = make_wepr(tmp_path, dense_weights(2))
    with pytest.raises(Exception, match="(?i)type|beartype"):
        scorer.compute({0: [-0.1, -0.2]})
