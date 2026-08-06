from importlib import resources

import numpy as np
import pytest
from conftest import chat_payloads_of_fixed_width
from hypothesis import HealthCheck, given, settings

from artefactual.exceptions import UncalibratedModelError
from artefactual.scoring.base_detector import DEFAULT_K, BaseDetector, epr, wepr

# Load package weights
_DATA = resources.files("artefactual.data")
EPR_WEIGHTS = _DATA / "calibration_ministral.json"
WEPR_WEIGHTS = _DATA / "weights_mistral_small.json"

# The shipped calibrations are fit at DEFAULT_K, and the parser refuses anything narrower,
# so responses are drawn at or above that width rather than read from a fixed fixture.
responses = chat_payloads_of_fixed_width(min_ranks=DEFAULT_K, max_ranks=20)
drawn = settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)


def test_epr_returns_base_detector():
    assert isinstance(epr(str(EPR_WEIGHTS)), BaseDetector)


def test_wepr_returns_base_detector():
    assert isinstance(wepr(str(WEPR_WEIGHTS)), BaseDetector)


@pytest.mark.parametrize("factory", [epr, wepr])
def test_factory_without_weights_raises(factory):
    with pytest.raises(UncalibratedModelError):
        factory()


def test_epr_step_names():
    assert [name for name, _ in epr(str(EPR_WEIGHTS)).steps] == ["parser", "entropy", "classifier"]


def test_epr_entropy_reduction():
    assert epr(str(EPR_WEIGHTS)).named_steps["entropy"].reduction == "epr"


def test_wepr_entropy_reduction():
    assert wepr(str(WEPR_WEIGHTS)).named_steps["entropy"].reduction == "wepr"


def test_epr_with_pretrained_has_coef():
    clf = epr(str(EPR_WEIGHTS)).named_steps["classifier"]
    assert clf.coef_.shape == (1, 1)  # 1 class, 1 feature (mean_entropy)


def test_from_pretrained_epr():
    detector = BaseDetector.from_pretrained(str(EPR_WEIGHTS), reduction="epr")
    assert isinstance(detector, BaseDetector)
    assert detector.named_steps["classifier"].coef_ is not None


@drawn
@given(response=responses)
def test_predict_proba_output_shape(response):
    scores = epr(str(EPR_WEIGHTS)).predict_proba(response)
    assert scores.shape == (1, 2)  # 1 sequence, 2 classes


@drawn
@given(response=responses)
def test_predict_proba_valid_probabilities(response):
    scores = epr(str(EPR_WEIGHTS)).predict_proba(response)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.allclose(scores.sum(axis=1), 1.0)


@drawn
@given(response=responses)
def test_predict_token_proba_shape(response):
    token_scores = epr(str(EPR_WEIGHTS)).predict_token_proba(response)
    assert token_scores.shape[0] == 1  # 1 sequence
    assert token_scores.shape[2] == 1


@drawn
@given(response=responses)
def test_predict_token_proba_valid_scores(response):
    token_scores = epr(str(EPR_WEIGHTS)).predict_token_proba(response)
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
    # matches how the shipped calibrations were fit, so coefficients stay comparable
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
def test_asking_for_both_pretrained_and_trainable_is_rejected(factory):
    # the two are contradictory; silently preferring one would hide a config mistake
    with pytest.raises(ValueError, match="not both"):
        factory(str(EPR_WEIGHTS), trainable=True)


@pytest.mark.parametrize("factory", [epr, wepr])
def test_a_classifier_without_trainable_is_rejected(factory):
    from sklearn.linear_model import LogisticRegression

    with pytest.raises(ValueError, match="only applies with trainable=True"):
        factory(str(WEPR_WEIGHTS), classifier=LogisticRegression())


@pytest.mark.parametrize("factory", [epr, wepr])
def test_neither_weights_nor_trainable_still_raises(factory):
    """No silent fallback to an unfitted classifier.

    A config key that resolves to None must not hand back a detector that trains on the
    caller's data and emits probabilities no calibration backs.
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
