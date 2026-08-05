import numpy as np
import pytest
from conftest import write_json
from sklearn.utils.validation import check_is_fitted

from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression

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


def load_from(tmp_path, payload):
    """Build a detector from `payload` written to a real weights file on disk."""
    return PretrainedLogisticRegression.from_pretrained(str(write_json(tmp_path, "weights.json", payload)))


# Tests for EPR Configuration


def test_from_pretrained_epr(tmp_path):
    clf = load_from(tmp_path, EPR_WEIGHTS)

    # Assert basic class instantiation
    assert isinstance(clf, PretrainedLogisticRegression)

    # Assert correct parameters parsed
    assert clf.n_features_in_ == 1
    np.testing.assert_array_almost_equal(clf.coef_, np.array([[1.25]], dtype=np.float32))
    np.testing.assert_array_almost_equal(clf.intercept_, np.array([-0.5], dtype=np.float32))

    # Assert "already fitted" flags
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))
    np.testing.assert_array_equal(clf.n_iter_, np.array([0]))


# Tests for WEPR Configuration


def test_from_pretrained_wepr(tmp_path):
    clf = load_from(tmp_path, WEPR_WEIGHTS)

    # Assert correct parameters parsed (k=2, so 2*k = 4 features)
    assert clf.n_features_in_ == 4

    # Expected order: [mean_rank_1, mean_rank_2, max_rank_1, max_rank_2]
    expected_coef = np.array([[0.5, 0.3, -0.2, -0.1]], dtype=np.float32)
    np.testing.assert_array_almost_equal(clf.coef_, expected_coef)
    np.testing.assert_array_almost_equal(clf.intercept_, np.array([0.1], dtype=np.float32))


# Scikit-Learn Compatibility Tests


# Check that predict and predict_proba work without calling .fit() first
def test_sklearn_methods_work_without_fit(tmp_path):
    clf = load_from(tmp_path, EPR_WEIGHTS)

    # Using dummy input data matching n_features_in_ (shape: n_samples, n_features)
    x = np.array([[0.8], [0.2]], dtype=np.float32)

    preds = clf.predict(x)
    probs = clf.predict_proba(x)

    assert preds.shape == (2,)
    assert probs.shape == (2, 2)
    np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(2))


# Check that the model predicts the correct class based on the loaded weights
def test_predict_output_values(tmp_path):
    clf = load_from(tmp_path, EPR_WEIGHTS)

    # high entropy → positive logit → class 1, low entropy → class 0
    x = np.array([[5.0], [0.0]], dtype=np.float32)
    preds = clf.predict(x)

    assert preds[0] == 1
    assert preds[1] == 0


# Check that sklearn considers the model fitted after loading pretrained weights
def test_check_is_fitted(tmp_path):
    check_is_fitted(load_from(tmp_path, EPR_WEIGHTS))  # raises if not fitted


# Edge Cases


# Check that a missing intercept key in the weights file raises a KeyError
def test_missing_intercept_raises(tmp_path):
    with pytest.raises(KeyError):
        load_from(tmp_path, {"coefficients": {"mean_entropy": 1.0}})


# Check that unrecognized coefficient keys raise an error instead of silently failing
def test_unknown_coefficients_raises(tmp_path):
    with pytest.raises((KeyError, ValueError)):
        load_from(tmp_path, {"intercept": 0.0, "coefficients": {"unknown_key": 1.0}})
