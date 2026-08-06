"""scikit-learn contracts for the two transformers and the pretrained classifier.

`BaseDetector` is a `Pipeline`, so anything that breaks clone/get_params/tags breaks
`GridSearchCV`, `cross_val_score` and `Pipeline` construction. These are the checks
sklearn itself would run, restricted to the ones that make sense for estimators that
deliberately opt out of array validation.
"""

import numpy as np
import pytest
from conftest import chat_payloads_of_fixed_width, logprob_cubes
from hypothesis import given
from sklearn.base import clone

from artefactual.exceptions import EmptySequenceWarning
from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression

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


# --- PretrainedLogisticRegression ------------------------------------------------------


def test_weights_with_a_max_rank_but_no_mean_rank_are_rejected(tmp_path):
    # only mean_rank_* drives the k count, so this file would build a 0-feature classifier
    path = tmp_path / "w.json"
    path.write_text('{"intercept": 0.0, "coefficients": {"max_rank_1": 1.0}}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unrecognized weights format"):
        PretrainedLogisticRegression.from_pretrained(str(path))


def test_a_wepr_file_missing_a_max_rank_is_reported(tmp_path):
    # WEPR zero-fills the gap; the classifier indexes it directly and raises instead
    path = tmp_path / "w.json"
    path.write_text(
        '{"intercept": 0.0, "coefficients": {"mean_rank_1": 1.0, "mean_rank_2": 1.0, "max_rank_1": 1.0}}',
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        PretrainedLogisticRegression.from_pretrained(str(path))


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


def test_widening_k_would_dilute_the_epr_feature():
    """Why the parser refuses a narrow response instead of zero-filling it.

    EPR is a mean over ranks, so absent ranks contribute 0 but still count in the
    denominator: reducing the same data at a wider k scales the feature by exactly the
    ratio of the two. That is a wrong score, not a rescaled one, because a genuinely wider
    response is not diluted the same way -- which is why k is the parser's to enforce and
    not a free parameter.
    """
    narrow = EntropyTransformer(reduction="epr").transform(LOGPROBS[:, :, :3])
    wide = EntropyTransformer(reduction="epr").transform(
        np.pad(LOGPROBS[:, :, :3], ((0, 0), (0, 0), (0, 7)), constant_values=np.nan)
    )

    np.testing.assert_allclose(wide, narrow * 3 / 10, rtol=1e-6)


def test_the_epr_feature_is_the_mean_not_the_sum_over_ranks():
    """Guards the scale the shipped calibrations were fit at.

    The coefficient is named `mean_entropy` and was produced by averaging the per-rank
    contribution columns. Summing instead inflates the feature by k and saturates every
    calibrated probability, which is a silent failure -- the ranking still looks right.
    """
    contributions = EntropyTransformer().entropy_contributions(LOGPROBS)
    k = LOGPROBS.shape[-1]

    feature = EntropyTransformer(reduction="epr").transform(LOGPROBS)

    np.testing.assert_allclose(feature[:, 0], contributions.sum(axis=-1).mean(axis=-1) / k, rtol=1e-6)


def test_padded_sequences_still_warn():
    # an all-NaN token is padding, not a real zero-entropy one
    padded = np.full((1, 2, 3), np.nan)

    with pytest.warns(EmptySequenceWarning):
        features = EntropyTransformer(reduction="epr").transform(padded)

    assert not features.any()
