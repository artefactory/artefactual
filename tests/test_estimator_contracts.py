"""scikit-learn contracts for the two transformers and the pretrained classifier.

`BaseDetector` is a `Pipeline`, so anything that breaks clone/get_params/tags breaks
`GridSearchCV`, `cross_val_score` and `Pipeline` construction. These are the checks
sklearn itself would run, restricted to the ones that make sense for estimators that
deliberately opt out of array validation.
"""

import warnings

import numpy as np
import pytest
from conftest import chat_payloads_of_fixed_width, logprob_cubes
from hypothesis import given
from sklearn.base import clone

from artefactual.exceptions import EmptySequenceWarning
from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer

# Drawn wherever the test only needs "some valid input"; the few assertions below that
# pin an exact number keep their literals, because that is what they are checking.
cubes = logprob_cubes()
wide_payloads = chat_payloads_of_fixed_width(min_ranks=8, max_ranks=8)

# (n_sequences, n_tokens, k), descending along the rank axis
LOGPROBS = np.array([[[-0.1, -2.0, -3.0], [-0.5, -1.5, -4.0]]])


# --- get_params / clone ----------------------------------------------------------------


@pytest.mark.parametrize("reduction", ["epr", "wepr"])
def test_transformer_survives_a_clone(reduction):
    original = EntropyTransformer(reduction=reduction)
    assert clone(original).get_params() == original.get_params()


def test_clone_produces_an_independent_transformer():
    original = EntropyTransformer(reduction="epr")
    copy = clone(original)
    copy.set_params(reduction="wepr")

    assert original.reduction == "epr"


def test_a_callable_reduction_survives_a_clone():
    def reduction(x, axis):
        return np.nanmean(x, axis=axis)

    assert clone(EntropyTransformer(reduction=reduction)).reduction is reduction


def test_the_parser_exposes_only_k():
    assert LogProbParser().get_params() == {"k": None}


# --- fit is stateless ------------------------------------------------------------------


@pytest.mark.parametrize("estimator", [EntropyTransformer(), LogProbParser()])
@given(logprobs=cubes)
def test_fit_returns_self(estimator, logprobs):
    assert estimator.fit(logprobs) is estimator


@given(logprobs=cubes)
def test_transformer_fit_accepts_a_target(logprobs):
    # Pipeline.fit forwards y to every step
    assert EntropyTransformer().fit(logprobs, np.ones(len(logprobs))) is not None


@given(logprobs=cubes)
def test_transform_without_fit_matches_transform_after_fit(logprobs):
    transformer = EntropyTransformer(reduction="epr")
    before = transformer.transform(logprobs)
    after = transformer.fit(logprobs).transform(logprobs)

    np.testing.assert_allclose(before, after)


@given(logprobs=cubes)
def test_fit_transform_matches_transform(logprobs):
    transformer = EntropyTransformer(reduction="wepr")
    np.testing.assert_allclose(transformer.fit_transform(logprobs), transformer.transform(logprobs))


# --- tags ------------------------------------------------------------------------------


def test_transformer_declares_it_needs_no_fit():
    assert EntropyTransformer().__sklearn_tags__().requires_fit is False


def test_transformer_declares_nan_support():
    # it consumes the NaN padding LogProbParser emits; without this sklearn rejects the input
    assert EntropyTransformer().__sklearn_tags__().input_tags.allow_nan is True


def test_parser_opts_out_of_array_validation():
    tags = LogProbParser().__sklearn_tags__()
    assert tags.no_validation is True
    assert tags.requires_fit is False
    assert tags.input_tags.two_d_array is False


# --- reduction shapes ------------------------------------------------------------------


@given(logprobs=cubes)
def test_epr_reduction_yields_one_feature(logprobs):
    assert EntropyTransformer(reduction="epr").transform(logprobs).shape == (len(logprobs), 1)


@given(logprobs=cubes)
def test_wepr_reduction_yields_two_features_per_rank(logprobs):
    # mean branch and max branch are concatenated, so 2k columns
    features = EntropyTransformer(reduction="wepr").transform(logprobs)
    assert features.shape == (len(logprobs), 2 * logprobs.shape[2])


@given(logprobs=cubes)
def test_token_mode_keeps_the_token_axis(logprobs):
    tokens = EntropyTransformer(reduction="epr").transform_tokens(logprobs)
    assert tokens.shape[:2] == logprobs.shape[:2]


# --- degenerate sequences --------------------------------------------------------------


def test_a_fully_padded_sequence_warns_and_scores_at_baseline():
    # a token-less sequence is all NaN; it must be surfaced, not silently zeroed
    padded = np.full((1, 2, 3), np.nan)

    with pytest.warns(EmptySequenceWarning, match="empty"):
        features = EntropyTransformer(reduction="epr").transform(padded)

    assert not features.any()


def test_the_warning_counts_the_empty_sequences():
    padded = np.full((2, 1, 2), np.nan)

    with pytest.warns(EmptySequenceWarning, match="2 empty"):
        EntropyTransformer(reduction="epr").transform(padded)


def test_a_populated_sequence_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", EmptySequenceWarning)
        EntropyTransformer(reduction="epr").transform(LOGPROBS)


@pytest.mark.parametrize("reduction", ["epr", "wepr"])
def test_padding_reports_once_not_twice(reduction):
    """`EmptySequenceWarning` is the only warning a padded batch produces.

    numpy's "Mean of empty slice" and "All-NaN slice encountered" describe the same
    condition from a frame inside the transformer, so they are silenced at the reduction.
    Silencing them must not also swallow the warning that names the offending sequences.
    """
    padded = np.full((1, 2, 3), np.nan)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        EntropyTransformer(reduction=reduction).transform(padded)

    assert [w.category for w in caught] == [EmptySequenceWarning]


def test_padded_tokens_do_not_drag_the_epr_mean_down():
    """NaN rows are padding, not zero-entropy tokens.

    `_epr` restores NaN over fully-NaN tokens before the mean precisely so a short
    sequence in a padded batch is not penalised for its padding.
    """
    single = np.array([[[-0.5, -1.5]]])
    padded = np.array([[[-0.5, -1.5], [np.nan, np.nan]]])

    np.testing.assert_allclose(
        EntropyTransformer(reduction="epr").transform(padded),
        EntropyTransformer(reduction="epr").transform(single),
    )


# --- the k parameter -------------------------------------------------------------------
#
# `k` belongs to the parser alone. The transformer reduces over whatever width it is
# handed, because by then the parser has already guaranteed that width is the calibrated
# one -- see test_rank_width.py for the contract itself.


@given(logprobs=cubes)
def test_the_transformer_carries_no_rank_count(logprobs):
    assert "k" not in EntropyTransformer().get_params()
    features = EntropyTransformer(reduction="wepr").transform(logprobs)
    assert features.shape == (len(logprobs), 2 * logprobs.shape[2])


@pytest.mark.parametrize("k", [1, 3, 8])
@given(payload=wide_payloads)
def test_the_parser_pins_the_rank_axis(k, payload):
    # payloads are drawn 8 ranks wide, so these truncate rather than trip the narrowness check
    assert LogProbParser(k=k).transform(payload).shape[2] == k


def test_k_survives_a_clone():
    original = LogProbParser(k=8)
    assert clone(original).k == 8
    assert original.get_params()["k"] == 8


@given(payload=wide_payloads)
def test_k_is_not_mutated_by_transform(payload):
    parser = LogProbParser(k=8)
    parser.transform(payload)
    assert parser.k == 8


def test_zero_filling_understates_the_epr_feature():
    """Why the parser refuses a narrow response instead of zero-filling it.

    EPR sums the rank axis, so ranks the caller never fetched contribute nothing and the
    sum stops short of the entropy the distribution actually carries. Padding a 3-rank
    response out to 10 therefore reproduces its 3-rank score, not the score a genuine
    10-rank response would have earned -- the two are not comparable, which is why the
    rank count is the parser's to enforce.
    """
    narrow = EntropyTransformer(reduction="epr").transform(LOGPROBS[:, :, :3])
    padded = EntropyTransformer(reduction="epr").transform(
        np.pad(LOGPROBS[:, :, :3], ((0, 0), (0, 0), (0, 7)), constant_values=np.nan)
    )

    np.testing.assert_allclose(padded, narrow, rtol=1e-6)


def test_the_epr_feature_is_the_truncated_entropy():
    """Pins EPR to Eq. 3 and 6: mean over tokens of the sum over ranks.

    Averaging the rank axis instead would still rank identically -- it is a constant
    rescale -- so nothing downstream would object, and the feature would silently stop
    being an entropy. A uniform distribution over k ranks has entropy ln(k) exactly, which
    catches the reduction and the log base at once. See test_feature_extraction.py for the
    comparison against the released implementation.
    """
    contributions = EntropyTransformer().entropy_contributions(LOGPROBS)
    feature = EntropyTransformer(reduction="epr").transform(LOGPROBS)

    np.testing.assert_allclose(feature[:, 0], contributions.sum(axis=-1).mean(axis=-1), rtol=1e-6)

    for k in (2, 8, 15):
        uniform = np.full((1, 1, k), np.log(1 / k))
        epr = EntropyTransformer(reduction="epr").transform(uniform)
        np.testing.assert_allclose(epr[0, 0], np.log(k), rtol=1e-6)


def test_padded_sequences_still_warn():
    # an all-NaN token is padding, not a real zero-entropy one
    padded = np.full((1, 2, 3), np.nan)

    with pytest.warns(EmptySequenceWarning):
        features = EntropyTransformer(reduction="epr").transform(padded)

    assert not features.any()
