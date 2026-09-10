import numpy as np
import pytest
from conftest import chat_payloads_of_fixed_width, fitted_logistic, write_estimator
from hypothesis import HealthCheck, given, settings

from artefactual.scoring.base_detector import DEFAULT_K, BaseDetector, epr, wepr


@pytest.fixture(scope="session")
def epr_estimator_path(tmp_path_factory):
    """A one-feature detector on disk, standing in for a published EPR one.

    Written here rather than read out of the package: estimators are published, not
    bundled, so a test that loaded one would need the Hub.
    """
    return str(write_estimator(tmp_path_factory.mktemp("epr"), "model.skops", fitted_logistic(-2.9, [58.2])))


@pytest.fixture(scope="session")
def wepr_estimator_path(tmp_path_factory):
    """A `2 * DEFAULT_K` feature detector on disk, standing in for a published WEPR one."""
    coefficients = [0.5 - 0.05 * index for index in range(2 * DEFAULT_K)]
    return str(write_estimator(tmp_path_factory.mktemp("wepr"), "model.skops", fitted_logistic(-3.4, coefficients)))


# The estimators are fit at DEFAULT_K, and the parser refuses anything narrower,
# so responses are drawn at or above that width rather than read from a fixed fixture.
responses = chat_payloads_of_fixed_width(min_ranks=DEFAULT_K, max_ranks=20)
drawn = settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)


def test_epr_returns_base_detector(epr_estimator_path):
    assert isinstance(BaseDetector.from_pretrained(epr_estimator_path, "epr"), BaseDetector)


def test_wepr_returns_base_detector(wepr_estimator_path):
    assert isinstance(BaseDetector.from_pretrained(wepr_estimator_path, "wepr"), BaseDetector)


def test_epr_step_names(epr_estimator_path):
    assert [name for name, _ in BaseDetector.from_pretrained(epr_estimator_path, "epr").steps] == [
        "parser",
        "entropy",
        "classifier",
    ]


def test_epr_entropy_reduction(epr_estimator_path):
    assert BaseDetector.from_pretrained(epr_estimator_path, "epr").named_steps["entropy"].reduction == "epr"


def test_wepr_entropy_reduction(wepr_estimator_path):
    assert BaseDetector.from_pretrained(wepr_estimator_path, "wepr").named_steps["entropy"].reduction == "wepr"


def test_epr_with_pretrained_has_coef(epr_estimator_path):
    clf = BaseDetector.from_pretrained(epr_estimator_path, "epr").named_steps["classifier"]
    assert clf.coef_.shape == (1, 1)  # 1 class, 1 feature (mean_entropy)


def test_from_pretrained_epr(epr_estimator_path):
    detector = BaseDetector.from_pretrained(epr_estimator_path, reduction="epr")
    assert isinstance(detector, BaseDetector)
    assert detector.named_steps["classifier"].coef_ is not None


@drawn
@given(response=responses)
def test_predict_proba_output_shape(epr_estimator_path, response):
    scores = BaseDetector.from_pretrained(epr_estimator_path, "epr").predict_proba(response)
    assert scores.shape == (1, 2)  # 1 sequence, 2 classes


@drawn
@given(response=responses)
def test_predict_proba_valid_probabilities(epr_estimator_path, response):
    scores = BaseDetector.from_pretrained(epr_estimator_path, "epr").predict_proba(response)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.allclose(scores.sum(axis=1), 1.0)


@drawn
@given(response=responses)
def test_predict_token_proba_shape(epr_estimator_path, response):
    token_scores = BaseDetector.from_pretrained(epr_estimator_path, "epr").predict_token_proba(response)
    assert token_scores.shape[0] == 1  # 1 sequence
    assert token_scores.shape[2] == 1


@drawn
@given(response=responses)
def test_predict_token_proba_valid_scores(epr_estimator_path, response):
    token_scores = BaseDetector.from_pretrained(epr_estimator_path, "epr").predict_token_proba(response)
    valid = token_scores[~np.isnan(token_scores)]
    assert len(valid) > 0
    assert np.all(valid >= 0) and np.all(valid <= 1)


# --- the unfitted pipeline the factories return ---------------------------------------


def _chat(ranks, n_tokens=2):
    token = {"token": "t", "logprob": ranks[0], "top_logprobs": [{"logprob": r} for r in ranks]}
    return {"choices": [{"logprobs": {"content": [token] * n_tokens}}]}


def test_trainable_returns_an_unfitted_detector():
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    detector = epr(k=3)

    assert [name for name, _ in detector.steps] == ["parser", "entropy", "classifier"]
    with pytest.raises(NotFittedError):
        check_is_fitted(detector.named_steps["classifier"])


def test_trainable_defaults_to_an_unregularised_regression():
    # matches how the shipped estimators were fit, so coefficients stay comparable
    classifier = epr().named_steps["classifier"]

    assert classifier.C == np.inf
    assert classifier.max_iter == 1000


def test_trainable_accepts_a_custom_classifier():
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=2)
    assert epr(classifier=forest).named_steps["classifier"] is forest


def test_trainable_pins_the_rank_width():
    assert wepr(k=7).named_steps["parser"].k == 7


@pytest.mark.parametrize("factory", [epr, wepr])
def test_a_weights_identifier_is_not_accepted_positionally(factory, epr_estimator_path):
    """The factories build unfitted detectors; weights are loaded by `from_pretrained`.

    `k` is keyword-only so that a call written for the old signature fails here rather
    than binding a repository id to the rank count and failing inside the parser.
    """
    with pytest.raises(TypeError):
        factory(epr_estimator_path)


@pytest.mark.parametrize("reduction", ["epr", "wepr"])
def test_a_trained_detector_scores_like_a_pretrained_one(reduction):
    """fit() then predict_proba() must work on raw responses, end to end."""
    k = 3
    confident = [_chat([-0.001, -8.0, -9.0]) for _ in range(4)]
    uncertain = [_chat([-1.0, -1.1, -1.2]) for _ in range(4)]
    x = confident + uncertain
    y = np.array([0] * 4 + [1] * 4)

    detector = BaseDetector.trainable(reduction, k=k).fit(x, y)
    scores = detector.predict_proba(x)

    assert scores.shape == (8, 2)
    assert np.all((scores >= 0) & (scores <= 1))
    # the fit should separate the two groups it was handed
    assert scores[:4, 1].mean() < scores[4:, 1].mean()


def test_a_trained_epr_detector_yields_one_coefficient():
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    detector = epr(k=3).fit(x, np.array([0, 1]))

    assert detector.named_steps["classifier"].coef_.shape == (1, 1)


def test_a_trained_wepr_detector_yields_two_coefficients_per_rank():
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    detector = wepr(k=3).fit(x, np.array([0, 1]))

    assert detector.named_steps["classifier"].coef_.shape == (1, 6)
