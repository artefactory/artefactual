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
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from sklearn.utils.validation import check_is_fitted

from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression
from artefactual.utils.io import MODEL_CALIBRATION_MAP, MODEL_WEIGHT_MAP

drawn = settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)

# Coefficients are drawn, so assertions compare against the payload that was written
# rather than against a literal repeated in the test.
two_rank_weights = wepr_weights(min_k=2, max_k=2)


def coefficient_count(payload):
    return sum(1 for key in payload["coefficients"] if key.startswith("mean_rank_"))


def load(tmp_path, payload, **kwargs):
    return PretrainedLogisticRegression.from_pretrained(str(write_json(tmp_path, "w.json", payload)), **kwargs)


# --- EPR calibrations ------------------------------------------------------------------


@drawn
@given(payload=epr_calibration())
def test_an_epr_calibration_yields_a_single_feature(tmp_path, payload):
    clf = load(tmp_path, payload)

    assert clf.n_features_in_ == 1
    np.testing.assert_array_almost_equal(clf.coef_, [[payload["coefficients"]["mean_entropy"]]])
    np.testing.assert_array_almost_equal(clf.intercept_, [payload["intercept"]])


@drawn
@given(payload=epr_calibration(), k=st.integers(1, 30))
def test_an_epr_calibration_ignores_k(tmp_path, payload, k):
    # the calibration records no rank count, so k is the parser's business, not its own
    assert load(tmp_path, payload, k=k).n_features_in_ == 1


# --- WEPR weights ----------------------------------------------------------------------


@drawn
@given(payload=two_rank_weights)
def test_wepr_weights_yield_two_features_per_rank(tmp_path, payload):
    clf = load(tmp_path, payload)

    assert clf.n_features_in_ == 4
    # order is [mean_rank_1..k, max_rank_1..k]
    coefficients = payload["coefficients"]
    expected = [coefficients["mean_rank_1"], coefficients["mean_rank_2"]]
    expected += [coefficients["max_rank_1"], coefficients["max_rank_2"]]
    np.testing.assert_array_almost_equal(clf.coef_, [expected])
    np.testing.assert_array_almost_equal(clf.intercept_, [payload["intercept"]])


@drawn
@given(payload=wepr_weights())
def test_wepr_weights_matching_k_are_accepted(tmp_path, payload):
    file_k = coefficient_count(payload)
    assert load(tmp_path, payload, k=file_k).n_features_in_ == 2 * file_k


@drawn
@given(payload=wepr_weights(), k=st.integers(1, 30))
def test_wepr_weights_disagreeing_with_k_are_rejected(tmp_path, payload, k):
    """The coefficient vector is fixed at the calibration rank count.

    Silently accepting a mismatch is what produced the raw sklearn shape error downstream;
    the message has to name both numbers and what to do about it.
    """
    file_k = coefficient_count(payload)
    assume(file_k != k)

    with pytest.raises(ValueError, match=rf"cover {file_k} rank\(s\) but k={k} was requested"):
        load(tmp_path, payload, k=k)


@drawn
@given(payload=wepr_weights(), k=st.integers(1, 30))
def test_the_mismatch_message_suggests_the_file_rank_count(tmp_path, payload, k):
    file_k = coefficient_count(payload)
    assume(file_k != k)

    with pytest.raises(ValueError, match=f"pass k={file_k}"):
        load(tmp_path, payload, k=k)


# --- registry selection ----------------------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(MODEL_WEIGHT_MAP))
def test_the_weights_registry_resolves_to_wepr_weights(model_name):
    clf = PretrainedLogisticRegression.from_pretrained(model_name, registry="weights", k=15)
    assert clf.n_features_in_ == 30


@pytest.mark.parametrize("model_name", sorted(MODEL_CALIBRATION_MAP))
def test_the_calibration_registry_resolves_to_epr_calibrations(model_name):
    clf = PretrainedLogisticRegression.from_pretrained(model_name, registry="calibration", k=15)
    assert clf.n_features_in_ == 1


@drawn
@given(payload=wepr_weights())
def test_the_registry_defaults_to_weights(tmp_path, payload):
    # wepr() is the caller that omits it; a path bypasses the registry either way
    assert load(tmp_path, payload).n_features_in_ == 2 * coefficient_count(payload)


def test_an_unknown_registry_is_rejected():
    with pytest.raises(KeyError):
        PretrainedLogisticRegression.from_pretrained("whatever", registry="nonsense")


# --- behaves as a fitted sklearn classifier --------------------------------------------


@drawn
@given(payload=epr_calibration())
def test_the_instance_reports_as_fitted(tmp_path, payload):
    check_is_fitted(load(tmp_path, payload))  # raises if not


@drawn
@given(payload=epr_calibration(), x=st.lists(st.floats(0.0, 5.0), min_size=2, max_size=2))
def test_predict_and_predict_proba_work_without_fit(tmp_path, payload, x):
    clf = load(tmp_path, payload)
    x = np.array([[value] for value in x], dtype=np.float32)

    assert clf.predict(x).shape == (2,)
    probabilities = clf.predict_proba(x)
    assert probabilities.shape == (2, 2)
    np.testing.assert_array_almost_equal(probabilities.sum(axis=1), np.ones(2))


def test_the_decision_boundary_follows_the_loaded_weights(tmp_path):
    """Fixed coefficients on purpose: this asserts the exact labels, not a property.

    A drawn calibration only says "prediction agrees with the logit", which is true of any
    logistic regression. Pinning intercept and slope is what checks the *direction*: high
    entropy must mean class 1.
    """
    clf = load(tmp_path, {"intercept": -0.5, "coefficients": {"mean_entropy": 1.25}})

    predicted = clf.predict(np.array([[5.0], [0.0]], dtype=np.float32))

    np.testing.assert_array_equal(predicted, [1, 0])


@drawn
@given(payload=epr_calibration())
def test_zero_iterations_are_reported(tmp_path, payload):
    clf = load(tmp_path, payload)
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
