"""Responses in, probabilities out: `make_pipeline(LogProbParser(k), classifier)` with the array classifiers."""

import numpy as np
import pytest
from conftest import (
    chat_payloads_of_fixed_width,
    fitted_logistic,
    load_detector,
    rank_vectors,
    token_proba,
    write_estimator,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from artefactual.preprocessing import LogProbParser
from artefactual.scoring import EPR, WEPR
from artefactual.scoring.entropy_methods.pooling import EPRPooling, WEPRPooling

# The published detectors were fit at 15 candidates per token.
K = 15


def _chat(ranks, n_tokens=2):
    token = {"token": "t", "logprob": ranks[0], "top_logprobs": [{"logprob": r} for r in ranks]}
    return {"choices": [{"logprobs": {"content": [token] * n_tokens}}]}


@pytest.fixture(scope="session")
def epr_estimator_path(tmp_path_factory):
    """A one-feature detector on disk, standing in for a published EPR one."""
    return str(write_estimator(tmp_path_factory.mktemp("epr"), "model.skops", fitted_logistic(-2.9, [58.2])))


@pytest.fixture(scope="session")
def wepr_estimator_path(tmp_path_factory):
    """A `2 * K` feature detector on disk, standing in for a published WEPR one."""
    coefficients = [0.5 - 0.05 * index for index in range(2 * K)]
    return str(write_estimator(tmp_path_factory.mktemp("wepr"), "model.skops", fitted_logistic(-3.4, coefficients)))


# The estimators are fit at K, and the parser refuses anything narrower,
# so responses are drawn at or above that width.
responses = chat_payloads_of_fixed_width(min_ranks=K, max_ranks=20)
drawn = settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)


def test_a_loaded_detector_is_a_parser_in_front_of_the_classifier(epr_estimator_path):
    detector = load_detector(EPR, epr_estimator_path, k=15)

    assert type(detector[0]) is LogProbParser
    assert type(detector[-1]) is EPR


def test_epr_entropy_pooling(epr_estimator_path):
    detector = load_detector(EPR, epr_estimator_path, k=15)
    detector.predict_proba(_chat([-0.1] * 15))

    assert detector[-1].pipeline_[-2].pooling == EPRPooling()


def test_wepr_entropy_pooling(wepr_estimator_path):
    detector = load_detector(WEPR, wepr_estimator_path, k=15)
    detector.predict_proba(_chat([-0.1] * 15))

    assert detector[-1].pipeline_[-2].pooling == WEPRPooling()


def test_epr_with_pretrained_has_coef(epr_estimator_path):
    clf = load_detector(EPR, epr_estimator_path, k=15)[-1].estimator
    assert clf.coef_.shape == (1, 1)  # 1 class, 1 feature (mean_entropy)


@drawn
@given(response=responses)
def test_predict_proba_output_shape(epr_estimator_path, response):
    scores = load_detector(EPR, epr_estimator_path, k=15).predict_proba(response)
    assert scores.shape == (1, 2)  # 1 sequence, 2 classes


@drawn
@given(response=responses)
def test_predict_proba_valid_probabilities(epr_estimator_path, response):
    scores = load_detector(EPR, epr_estimator_path, k=15).predict_proba(response)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.allclose(scores.sum(axis=1), 1.0)


@drawn
@given(response=responses)
def test_predict_token_proba_shape(epr_estimator_path, response):
    token_scores = token_proba(load_detector(EPR, epr_estimator_path, k=15), response)
    assert token_scores.shape[0] == 1  # 1 sequence
    assert token_scores.shape[2] == 1


@drawn
@given(response=responses)
def test_predict_token_proba_valid_scores(epr_estimator_path, response):
    token_scores = token_proba(load_detector(EPR, epr_estimator_path, k=15), response)
    valid = token_scores[~np.isnan(token_scores)]
    assert len(valid) > 0
    assert np.all(valid >= 0) and np.all(valid <= 1)


# --- the unfitted pipeline ------------------------------------------------------------


def detector_of(cls, k, **options):
    """An unfitted responses -> probabilities pipeline over `cls`."""
    return make_pipeline(LogProbParser(k=k), cls(**options))


def test_a_constructed_detector_is_unfitted():
    with pytest.raises(NotFittedError):
        check_is_fitted(detector_of(EPR, 3)[-1])


def test_a_detector_defaults_to_an_unregularised_regression():
    # matches how the shipped estimators were fit, so coefficients stay comparable
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    classifier = detector_of(EPR, 3).fit(x, np.array([0, 1]))[-1].pipeline_[-1]

    assert classifier.C == np.inf
    assert classifier.max_iter == 1000


def test_a_detector_accepts_a_custom_estimator_and_never_fits_it_in_place():
    forest = RandomForestClassifier(n_estimators=2)
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    detector = detector_of(EPR, 3, estimator=forest).fit(x, np.array([0, 1]))

    assert detector[-1].estimator is forest
    assert type(detector[-1].pipeline_[-1]) is RandomForestClassifier
    with pytest.raises(NotFittedError):
        check_is_fitted(forest)


def test_the_parser_and_the_classifier_agree_on_the_rank_width():
    x = [_chat([-1.0, -1.1, -1.2, -1.3])] * 2
    detector = make_pipeline(LogProbParser(k=4), EPR(k=3))

    with pytest.raises(ValueError, match="trained on 3 candidates per token, got 4"):
        detector.fit(x, np.array([0, 1]))


@pytest.mark.parametrize("detector_class", [EPR, WEPR])
def test_a_pipeline_trained_on_responses_scores_them(detector_class):
    """fit() then predict_proba() must work on raw responses, end to end."""
    k = 3
    confident = [_chat([-0.001, -8.0, -9.0]) for _ in range(4)]
    uncertain = [_chat([-1.0, -1.1, -1.2]) for _ in range(4)]
    x = confident + uncertain
    y = np.array([0] * 4 + [1] * 4)

    detector = detector_of(detector_class, k).fit(x, y)
    scores = detector.predict_proba(x)

    assert scores.shape == (8, 2)
    assert np.all((scores >= 0) & (scores <= 1))
    # the fit should separate the two groups it was handed
    assert scores[:4, 1].mean() < scores[4:, 1].mean()


def test_a_trained_epr_detector_yields_one_coefficient():
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    detector = detector_of(EPR, 3).fit(x, np.array([0, 1]))

    assert detector[-1].pipeline_[-1].coef_.shape == (1, 1)


def test_a_trained_wepr_detector_yields_two_coefficients_per_rank():
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    detector = detector_of(WEPR, 3).fit(x, np.array([0, 1]))

    assert detector[-1].pipeline_[-1].coef_.shape == (1, 6)


# --- token padding ---------------------------------------------------------------------


@st.composite
def ragged_batches(draw, max_tokens=6):
    """Payloads of `K` candidates per token whose token counts differ, from one token up."""
    lengths = draw(st.lists(st.integers(1, max_tokens), min_size=1, max_size=4))
    return [
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"logprob": value} for value in draw(rank_vectors(min_ranks=K, max_ranks=K))
                                ]
                            }
                            for _ in range(length)
                        ]
                    }
                }
            ]
        }
        for length in lengths
    ], lengths


@drawn
@given(batch=ragged_batches())
def test_token_scores_are_padded_to_the_longest_sequence_with_nan(epr_estimator_path, batch):
    payloads, lengths = batch
    scores = token_proba(load_detector(EPR, epr_estimator_path, k=K), payloads)
    assert scores.shape == (len(lengths), max(lengths), 1)
    for row, length in zip(scores[..., 0], lengths, strict=True):
        assert not np.isnan(row[:length]).any()
        assert np.isnan(row[length:]).all()
        assert np.all((row[:length] >= 0) & (row[:length] <= 1))


@drawn
@given(batch=ragged_batches())
def test_padding_does_not_change_the_score_of_a_sequence(epr_estimator_path, batch):
    """A sequence scores the same alone as inside a batch, whatever its neighbours' length."""
    payloads, lengths = batch
    detector = load_detector(EPR, epr_estimator_path, k=K)
    together = token_proba(detector, payloads)
    for index, (payload, length) in enumerate(zip(payloads, lengths, strict=True)):
        np.testing.assert_allclose(together[index, :length], token_proba(detector, payload)[0], rtol=1e-6)


def test_an_empty_batch_has_no_token_scores(epr_estimator_path):
    scores = token_proba(load_detector(EPR, epr_estimator_path, k=K), [])
    assert scores.shape[0] == 0
