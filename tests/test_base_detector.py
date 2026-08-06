import numpy as np
import pytest
from conftest import chat_payloads_of_fixed_width, fitted_logistic, write_estimator
from hypothesis import HealthCheck, given, settings

from artefactual.exceptions import UncalibratedModelError
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
    assert isinstance(epr(epr_estimator_path), BaseDetector)


def test_wepr_returns_base_detector(wepr_estimator_path):
    assert isinstance(wepr(wepr_estimator_path), BaseDetector)


@pytest.mark.parametrize("factory", [epr, wepr])
def test_factory_without_weights_raises(factory):
    with pytest.raises(UncalibratedModelError):
        factory()


def test_epr_step_names(epr_estimator_path):
    assert [name for name, _ in epr(epr_estimator_path).steps] == ["parser", "entropy", "classifier"]


def test_epr_entropy_reduction(epr_estimator_path):
    assert epr(epr_estimator_path).named_steps["entropy"].reduction == "epr"


def test_wepr_entropy_reduction(wepr_estimator_path):
    assert wepr(wepr_estimator_path).named_steps["entropy"].reduction == "wepr"


def test_epr_with_pretrained_has_coef(epr_estimator_path):
    clf = epr(epr_estimator_path).named_steps["classifier"]
    assert clf.coef_.shape == (1, 1)  # 1 class, 1 feature (mean_entropy)


def test_from_pretrained_epr(epr_estimator_path):
    detector = BaseDetector.from_pretrained(epr_estimator_path, reduction="epr")
    assert isinstance(detector, BaseDetector)
    assert detector.named_steps["classifier"].coef_ is not None


@drawn
@given(response=responses)
def test_predict_proba_output_shape(epr_estimator_path, response):
    scores = epr(epr_estimator_path).predict_proba(response)
    assert scores.shape == (1, 2)  # 1 sequence, 2 classes


@drawn
@given(response=responses)
def test_predict_proba_valid_probabilities(epr_estimator_path, response):
    scores = epr(epr_estimator_path).predict_proba(response)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.allclose(scores.sum(axis=1), 1.0)


@drawn
@given(response=responses)
def test_predict_token_proba_shape(epr_estimator_path, response):
    token_scores = epr(epr_estimator_path).predict_token_proba(response)
    assert token_scores.shape[0] == 1  # 1 sequence
    assert token_scores.shape[2] == 1


@drawn
@given(response=responses)
def test_predict_token_proba_valid_scores(epr_estimator_path, response):
    token_scores = epr(epr_estimator_path).predict_token_proba(response)
    valid = token_scores[~np.isnan(token_scores)]
    assert len(valid) > 0
    assert np.all(valid >= 0) and np.all(valid <= 1)


# --- trainable=True: the unfitted pipeline for calibrating on your own data -----------


def _chat(ranks, n_tokens=2):
    token = {"token": "t", "logprob": ranks[0], "top_logprobs": [{"logprob": r} for r in ranks]}
    return {"choices": [{"logprobs": {"content": [token] * n_tokens}}]}


def test_trainable_returns_an_unfitted_detector():
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    detector = epr(k=3, trainable=True)

    assert [name for name, _ in detector.steps] == ["parser", "entropy", "classifier"]
    with pytest.raises(NotFittedError):
        check_is_fitted(detector.named_steps["classifier"])


def test_trainable_defaults_to_an_unregularised_regression():
    # matches how the shipped estimators were fit, so coefficients stay comparable
    classifier = epr(trainable=True).named_steps["classifier"]

    assert classifier.C == np.inf
    assert classifier.max_iter == 1000


def test_trainable_accepts_a_custom_classifier():
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=2)
    assert epr(trainable=True, classifier=forest).named_steps["classifier"] is forest


def test_trainable_pins_the_rank_width():
    assert wepr(k=7, trainable=True).named_steps["parser"].k == 7


@pytest.mark.parametrize("factory", [epr, wepr])
def test_asking_for_both_pretrained_and_trainable_is_rejected(epr_estimator_path, factory):
    # the two are contradictory; silently preferring one would hide a config mistake
    with pytest.raises(ValueError, match="not both"):
        factory(epr_estimator_path, trainable=True)


@pytest.mark.parametrize("factory", [epr, wepr])
def test_a_classifier_without_trainable_is_rejected(wepr_estimator_path, factory):
    from sklearn.linear_model import LogisticRegression

    with pytest.raises(ValueError, match="only applies with trainable=True"):
        factory(wepr_estimator_path, classifier=LogisticRegression())


@pytest.mark.parametrize("factory", [epr, wepr])
def test_neither_weights_nor_trainable_still_raises(factory):
    """No silent fallback to an unfitted classifier.

    A config key that resolves to None must not hand back a detector that trains on the
    caller's data and emits probabilities no detector backs.
    """
    with pytest.raises(UncalibratedModelError, match="trainable"):
        factory()


@pytest.mark.parametrize("reduction", ["epr", "wepr"])
def test_a_trained_detector_scores_like_a_pretrained_one(reduction):
    """fit() then predict_proba() must work on raw responses, end to end."""
    k = 3
    confident = [_chat([-0.001, -8.0, -9.0]) for _ in range(4)]
    uncertain = [_chat([-1.0, -1.1, -1.2]) for _ in range(4)]
    x = confident + uncertain
    y = np.array([0] * 4 + [1] * 4)

    detector = {"epr": epr, "wepr": wepr}[reduction](k=k, trainable=True).fit(x, y)
    scores = detector.predict_proba(x)

    assert scores.shape == (8, 2)
    assert np.all((scores >= 0) & (scores <= 1))
    # the fit should separate the two groups it was handed
    assert scores[:4, 1].mean() < scores[4:, 1].mean()


def test_a_trained_epr_detector_yields_one_coefficient():
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    detector = epr(k=3, trainable=True).fit(x, np.array([0, 1]))

    assert detector.named_steps["classifier"].coef_.shape == (1, 1)


def test_a_trained_wepr_detector_yields_two_coefficients_per_rank():
    x = [_chat([-0.001, -8.0, -9.0]), _chat([-1.0, -1.1, -1.2])]
    detector = wepr(k=3, trainable=True).fit(x, np.array([0, 1]))

    assert detector.named_steps["classifier"].coef_.shape == (1, 6)
