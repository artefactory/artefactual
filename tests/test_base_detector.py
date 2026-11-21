import numpy as np
import pytest
from conftest import chat_payloads_of_fixed_width, fitted_logistic, rank_vectors, write_estimator
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from artefactual.scoring.base_detector import EPR, WEPR, BaseDetector
from artefactual.scoring.entropy_methods.pooling import EPRPooling, WEPRPooling

# The published detectors were fit at 15 candidates per token.
K = 15


@pytest.fixture(scope="session")
def epr_estimator_path(tmp_path_factory):
    """A one-feature detector on disk, standing in for a published EPR one.

    Written here rather than read out of the package: estimators are published, not
    bundled, so a test that loaded one would need the Hub.
    """
    return str(write_estimator(tmp_path_factory.mktemp("epr"), "model.skops", fitted_logistic(-2.9, [58.2])))


@pytest.fixture(scope="session")
def wepr_estimator_path(tmp_path_factory):
    """A `2 * K` feature detector on disk, standing in for a published WEPR one."""
    coefficients = [0.5 - 0.05 * index for index in range(2 * K)]
    return str(write_estimator(tmp_path_factory.mktemp("wepr"), "model.skops", fitted_logistic(-3.4, coefficients)))


# The estimators are fit at K, and the parser refuses anything narrower,
# so responses are drawn at or above that width rather than read from a fixed fixture.
responses = chat_payloads_of_fixed_width(min_ranks=K, max_ranks=20)
drawn = settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)


def test_epr_returns_base_detector(epr_estimator_path):
    assert isinstance(EPR.from_pretrained(epr_estimator_path, k=15), BaseDetector)


def test_wepr_returns_base_detector(wepr_estimator_path):
    assert isinstance(WEPR.from_pretrained(wepr_estimator_path, k=15), BaseDetector)


def test_epr_step_names(epr_estimator_path):
    assert [name for name, _ in EPR.from_pretrained(epr_estimator_path, k=15).steps] == [
        "parser",
        "entropy",
        "classifier",
    ]


def test_epr_entropy_pooling(epr_estimator_path):
    assert EPR.from_pretrained(epr_estimator_path, k=15).named_steps["entropy"].pooling == EPRPooling()


def test_wepr_entropy_pooling(wepr_estimator_path):
    assert WEPR.from_pretrained(wepr_estimator_path, k=15).named_steps["entropy"].pooling == WEPRPooling()


def test_epr_with_pretrained_has_coef(epr_estimator_path):
    clf = EPR.from_pretrained(epr_estimator_path, k=15).named_steps["classifier"]
    assert clf.coef_.shape == (1, 1)  # 1 class, 1 feature (mean_entropy)


def test_from_pretrained_epr(epr_estimator_path):
    detector = EPR.from_pretrained(epr_estimator_path, k=15)
    assert isinstance(detector, BaseDetector)
    assert detector.named_steps["classifier"].coef_ is not None


@drawn
@given(response=responses)
def test_predict_proba_output_shape(epr_estimator_path, response):
    scores = EPR.from_pretrained(epr_estimator_path, k=15).predict_proba(response)
    assert scores.shape == (1, 2)  # 1 sequence, 2 classes


@drawn
@given(response=responses)
def test_predict_proba_valid_probabilities(epr_estimator_path, response):
    scores = EPR.from_pretrained(epr_estimator_path, k=15).predict_proba(response)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.allclose(scores.sum(axis=1), 1.0)


@drawn
@given(response=responses)
def test_predict_token_proba_shape(epr_estimator_path, response):
    token_scores = EPR.from_pretrained(epr_estimator_path, k=15).predict_token_proba(response)
    assert token_scores.shape[0] == 1  # 1 sequence
    assert token_scores.shape[2] == 1


@drawn
@given(response=responses)
def test_predict_token_proba_valid_scores(epr_estimator_path, response):
    token_scores = EPR.from_pretrained(epr_estimator_path, k=15).predict_token_proba(response)
    valid = token_scores[~np.isnan(token_scores)]
    assert len(valid) > 0
    assert np.all(valid >= 0) and np.all(valid <= 1)


# --- the unfitted pipeline the factories return ---------------------------------------


def _chat(ranks, n_tokens=2):
    token = {"token": "t", "logprob": ranks[0], "top_logprobs": [{"logprob": r} for r in ranks]}
    return {"choices": [{"logprobs": {"content": [token] * n_tokens}}]}


def test_a_constructed_detector_is_unfitted():
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    detector = EPR(k=3)

    assert [name for name, _ in detector.steps] == ["parser", "entropy", "classifier"]
    with pytest.raises(NotFittedError):
        check_is_fitted(detector.named_steps["classifier"])


def test_a_detector_defaults_to_an_unregularised_regression():
    # matches how the shipped estimators were fit, so coefficients stay comparable
    classifier = EPR(k=15).named_steps["classifier"]

    assert classifier.C == np.inf
    assert classifier.max_iter == 1000


def test_a_detector_accepts_a_custom_estimator():
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=2)
    assert EPR(estimator=forest, k=15).named_steps["classifier"] is forest


def test_a_detector_pins_the_rank_width():
    assert WEPR(k=7).named_steps["parser"].k == 7


@pytest.mark.parametrize("detector_class", [EPR, WEPR])
def test_a_weights_identifier_is_not_accepted_as_a_rank_count(detector_class, epr_estimator_path):
    """Constructing a detector never loads weights; `from_pretrained` is what does.

    The first constructor argument is `k`, so a call written for the old factory signature
    hands a repository id to the parser as a rank count. It is refused when the pipeline
    runs rather than scoring against a width nothing checked.
    """
    detector = detector_class(epr_estimator_path)

    assert detector.k == epr_estimator_path
    with pytest.raises((TypeError, ValueError)):
        detector.predict_proba([_chat([-0.1, -0.2, -0.3])])


@pytest.mark.parametrize("detector_class", [EPR, WEPR])
def test_a_trained_detector_scores_like_a_pretrained_one(detector_class):
    """fit() then predict_proba() must work on raw responses, end to end."""
    k = 3
    confident = [_chat([-0.001, -8.0, -9.0]) for _ in range(4)]
    uncertain = [_chat([-1.0, -1.1, -1.2]) for _ in range(4)]
    x = confident + uncertain
    y = np.array([0] * 4 + [1] * 4)

    detector = detector_class(k=k).fit(x, y)
    scores = detector.predict_proba(x)

    assert scores.shape == (8, 2)
    assert np.all((scores >= 0) & (scores <= 1))
    # the fit should separate the two groups it was handed
    assert scores[:4, 1].mean() < scores[4:, 1].mean()


def test_a_trained_epr_detector_yields_one_coefficient():
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    detector = EPR(k=3).fit(x, np.array([0, 1]))

    assert detector.named_steps["classifier"].coef_.shape == (1, 1)


def test_a_trained_wepr_detector_yields_two_coefficients_per_rank():
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    detector = WEPR(k=3).fit(x, np.array([0, 1]))

    assert detector.named_steps["classifier"].coef_.shape == (1, 6)


# --- token-mode dispatch ---------------------------------------------------------------
#
# `predict_token_proba` routes each transformer through `transform_tokens` where the step
# offers one and `transform` otherwise. The capability has to be read off the step before
# anything is called: deciding it from whether a call raised `AttributeError` cannot tell
# a step that lacks the method from one whose method raised internally, and the second
# case silently produces sequence-reduced data in the token path.


class _TokenAware:
    """A step that reduces over the token axis in `transform` and keeps it in token mode."""

    def transform(self, x):
        return np.nanmean(x, axis=1)

    def transform_tokens(self, x):
        return x


class _TokenBlind:
    """A step with no token mode, which must be driven through `transform`."""

    def transform(self, x):
        return x


class _BrokenTokenMode:
    """A step whose token mode is broken by a bug of its own, not by being absent."""

    def transform(self, x):
        return np.nanmean(x, axis=1)

    def transform_tokens(self, x):  # ruff: ignore[unused-method-argument] — the bug is the raise, not the argument
        msg = "this step's own bug, not a missing method"
        raise AttributeError(msg)


def _detector(step):
    """A detector whose transformer steps are stubs, to drive the token-mode dispatch alone.

    Built as an `EPR` and then re-stepped: what is under test is how `predict_token_proba`
    drives whatever steps it finds, not the parser and entropy steps a detector assembles
    for itself.
    """
    detector = EPR(k=1)
    detector.steps = [("step", step), ("classifier", fitted_logistic(-1.0, [1.0]))]
    return detector


TOKEN_FEATURES = np.array([[[0.1], [0.2], [0.3]]])


def test_a_step_with_a_token_mode_is_driven_through_it():
    scores = _detector(_TokenAware()).predict_token_proba(TOKEN_FEATURES)
    assert scores.shape == (1, 3, 1)


def test_a_step_without_a_token_mode_falls_back_to_transform():
    scores = _detector(_TokenBlind()).predict_token_proba(TOKEN_FEATURES)
    assert scores.shape == (1, 3, 1)


def test_a_broken_token_mode_is_reported_not_swallowed():
    # The bug this pins: an `AttributeError` raised *inside* `transform_tokens` used to be
    # read as "this step has no token mode", rerouting to `transform` and reducing the
    # token axis away. The failure then surfaced far downstream as a shape error, or not
    # at all.
    with pytest.raises(AttributeError, match="this step's own bug"):
        _detector(_BrokenTokenMode()).predict_token_proba(TOKEN_FEATURES)


class _ReducingTokenBlind:
    """A step that reduces the token axis away and declares no token mode."""

    def transform(self, x):
        return np.nanmean(x, axis=1)


def test_a_step_that_reduces_the_token_axis_away_is_named():
    with pytest.raises(ValueError, match="declares no transform_tokens"):
        _detector(_ReducingTokenBlind()).predict_token_proba(TOKEN_FEATURES)


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
    scores = EPR.from_pretrained(epr_estimator_path, k=K).predict_token_proba(payloads)
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
    detector = EPR.from_pretrained(epr_estimator_path, k=K)
    together = detector.predict_token_proba(payloads)
    for index, (payload, length) in enumerate(zip(payloads, lengths, strict=True)):
        np.testing.assert_allclose(together[index, :length], detector.predict_token_proba(payload)[0], rtol=1e-6)


def test_an_empty_batch_has_no_token_scores(epr_estimator_path):
    scores = EPR.from_pretrained(epr_estimator_path, k=K).predict_token_proba([])
    assert scores.shape[0] == 0
