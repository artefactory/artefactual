"""Tests for the classifier that loads calibrated coefficients instead of fitting.

Weights are written to real JSON files rather than patched over the loader, so the
registry/path resolution in `utils.io` is exercised on the same path production uses.

Two things the class has to get right: the *metric* is inferred from the coefficient keys
(`mean_entropy` -> EPR, `mean_rank_1` -> WEPR), while the *registry* a bare model name
resolves through cannot be inferred — both registries are keyed by the same names — so the
caller states it.
"""

import numpy as np
import pytest
from conftest import epr_calibration, wepr_weights, write_json
from hypothesis import HealthCheck, given, settings
from sklearn.utils.validation import check_is_fitted

from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression
from artefactual.utils.io import MODEL_CALIBRATION_MAP, MODEL_WEIGHT_MAP

EPR_WEIGHTS = {"intercept": -0.5, "coefficients": {"mean_entropy": 1.25}}
WEPR_WEIGHTS = {
    "intercept": 0.1,
    "coefficients": {
        "mean_rank_1": 0.5,
        "mean_rank_2": 0.3,
        "max_rank_1": -0.2,
        "max_rank_2": -0.1,
    },
}


def load(tmp_path, payload, **kwargs):
    return PretrainedLogisticRegression.from_pretrained(str(write_json(tmp_path, "w.json", payload)), **kwargs)


# --- EPR calibrations ------------------------------------------------------------------


def test_an_epr_calibration_yields_a_single_feature(tmp_path):
    clf = load(tmp_path, EPR_WEIGHTS)

    assert clf.n_features_in_ == 1
    np.testing.assert_array_almost_equal(clf.coef_, [[1.25]])
    np.testing.assert_array_almost_equal(clf.intercept_, [-0.5])


def test_an_epr_calibration_ignores_k(tmp_path):
    # the calibration records no rank count, so k only governs alignment upstream
    assert load(tmp_path, EPR_WEIGHTS, k=15).n_features_in_ == 1


# --- WEPR weights ----------------------------------------------------------------------


def test_wepr_weights_yield_two_features_per_rank(tmp_path):
    clf = load(tmp_path, WEPR_WEIGHTS)

    assert clf.n_features_in_ == 4
    # order is [mean_rank_1..k, max_rank_1..k]
    np.testing.assert_array_almost_equal(clf.coef_, [[0.5, 0.3, -0.2, -0.1]])
    np.testing.assert_array_almost_equal(clf.intercept_, [0.1])


def test_wepr_weights_matching_k_are_accepted(tmp_path):
    assert load(tmp_path, WEPR_WEIGHTS, k=2).n_features_in_ == 4


@pytest.mark.parametrize("k", [1, 3, 15])
def test_wepr_weights_disagreeing_with_k_are_rejected(tmp_path, k):
    """The coefficient vector is fixed at the calibration rank count.

    Silently accepting a mismatch is what produced the raw sklearn shape error downstream;
    the message has to name both numbers and what to do about it.
    """
    with pytest.raises(ValueError, match=rf"cover 2 rank\(s\) but k={k} was requested"):
        load(tmp_path, WEPR_WEIGHTS, k=k)


def test_the_mismatch_message_suggests_the_file_rank_count(tmp_path):
    with pytest.raises(ValueError, match="pass k=2"):
        load(tmp_path, WEPR_WEIGHTS, k=15)


# --- registry selection ----------------------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(MODEL_WEIGHT_MAP))
def test_the_weights_registry_resolves_to_wepr_weights(model_name):
    clf = PretrainedLogisticRegression.from_pretrained(model_name, registry="weights", k=15)
    assert clf.n_features_in_ == 30


@pytest.mark.parametrize("model_name", sorted(MODEL_CALIBRATION_MAP))
def test_the_calibration_registry_resolves_to_epr_calibrations(model_name):
    clf = PretrainedLogisticRegression.from_pretrained(model_name, registry="calibration", k=15)
    assert clf.n_features_in_ == 1


def test_the_registry_defaults_to_weights(tmp_path):
    # wepr() is the caller that omits it; a path bypasses the registry either way
    assert load(tmp_path, WEPR_WEIGHTS).n_features_in_ == 4


def test_an_unknown_registry_is_rejected():
    with pytest.raises(KeyError):
        PretrainedLogisticRegression.from_pretrained("whatever", registry="nonsense")


# --- behaves as a fitted sklearn classifier --------------------------------------------


def test_the_instance_reports_as_fitted(tmp_path):
    check_is_fitted(load(tmp_path, EPR_WEIGHTS))  # raises if not


def test_predict_and_predict_proba_work_without_fit(tmp_path):
    clf = load(tmp_path, EPR_WEIGHTS)
    x = np.array([[0.8], [0.2]], dtype=np.float32)

    assert clf.predict(x).shape == (2,)
    probabilities = clf.predict_proba(x)
    assert probabilities.shape == (2, 2)
    np.testing.assert_array_almost_equal(probabilities.sum(axis=1), np.ones(2))


def test_the_decision_boundary_follows_the_loaded_weights(tmp_path):
    # high entropy -> positive logit -> class 1; low entropy -> class 0
    clf = load(tmp_path, EPR_WEIGHTS)
    np.testing.assert_array_equal(clf.predict(np.array([[5.0], [0.0]], dtype=np.float32)), [1, 0])


def test_zero_iterations_are_reported(tmp_path):
    clf = load(tmp_path, EPR_WEIGHTS)
    np.testing.assert_array_equal(clf.classes_, [0, 1])
    np.testing.assert_array_equal(clf.n_iter_, [0])


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(payload=wepr_weights())
def test_any_dense_wepr_file_loads(tmp_path, payload):
    k = sum(1 for key in payload["coefficients"] if key.startswith("mean_rank_"))
    assert load(tmp_path, payload, k=k).n_features_in_ == 2 * k


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(payload=epr_calibration())
def test_any_epr_calibration_loads(tmp_path, payload):
    assert load(tmp_path, payload).n_features_in_ == 1


# --- malformed files -------------------------------------------------------------------


def test_a_missing_intercept_is_reported(tmp_path):
    with pytest.raises(KeyError):
        load(tmp_path, {"coefficients": {"mean_entropy": 1.0}})


def test_unrecognised_coefficients_name_what_was_found(tmp_path):
    with pytest.raises(ValueError, match="Unrecognized weights format"):
        load(tmp_path, {"intercept": 0.0, "coefficients": {"unknown_key": 1.0}})


def test_max_rank_without_mean_rank_is_rejected(tmp_path):
    # only mean_rank_* drives the count, so this would otherwise build 0 features
    with pytest.raises(ValueError, match="Unrecognized weights format"):
        load(tmp_path, {"intercept": 0.0, "coefficients": {"max_rank_1": 1.0}})


def test_a_wepr_file_missing_a_max_rank_is_reported(tmp_path):
    payload = {"intercept": 0.0, "coefficients": {"mean_rank_1": 1.0, "mean_rank_2": 1.0, "max_rank_1": 1.0}}
    with pytest.raises(KeyError):
        load(tmp_path, payload)
