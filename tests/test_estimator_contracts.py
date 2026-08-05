"""scikit-learn contracts for the two transformers and the pretrained classifier.

`BaseDetector` is a `Pipeline`, so anything that breaks clone/get_params/tags breaks
`GridSearchCV`, `cross_val_score` and `Pipeline` construction. These are the checks
sklearn itself would run, restricted to the ones that make sense for estimators that
deliberately opt out of array validation.
"""

import numpy as np
import pytest
from sklearn.base import clone

from artefactual.exceptions import EmptySequenceWarning
from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression

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


def test_parser_exposes_no_hyperparameters():
    assert LogProbParser().get_params() == {}


# --- fit is stateless ------------------------------------------------------------------


@pytest.mark.parametrize("estimator", [EntropyTransformer(), LogProbParser()])
def test_fit_returns_self(estimator):
    assert estimator.fit(LOGPROBS) is estimator


def test_transformer_fit_accepts_a_target():
    # Pipeline.fit forwards y to every step
    assert EntropyTransformer().fit(LOGPROBS, np.array([1])) is not None


def test_transform_without_fit_matches_transform_after_fit():
    transformer = EntropyTransformer(reduction="epr")
    before = transformer.transform(LOGPROBS)
    after = transformer.fit(LOGPROBS).transform(LOGPROBS)

    np.testing.assert_allclose(before, after)


def test_fit_transform_matches_transform():
    transformer = EntropyTransformer(reduction="wepr")
    np.testing.assert_allclose(transformer.fit_transform(LOGPROBS), transformer.transform(LOGPROBS))


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


def test_epr_reduction_yields_one_feature():
    assert EntropyTransformer(reduction="epr").transform(LOGPROBS).shape == (1, 1)


def test_wepr_reduction_yields_two_features_per_rank():
    # mean branch and max branch are concatenated, so 2k columns
    assert EntropyTransformer(reduction="wepr").transform(LOGPROBS).shape == (1, 2 * LOGPROBS.shape[2])


def test_token_mode_keeps_the_token_axis():
    tokens = EntropyTransformer(reduction="epr").transform_tokens(LOGPROBS)
    assert tokens.shape[:2] == LOGPROBS.shape[:2]


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
