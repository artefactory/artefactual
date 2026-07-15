from unittest.mock import patch

import numpy as np
import pytest

from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression


@pytest.fixture
def mock_epr_weights():
    return {"intercept": -0.5, "coefficients": {"mean_entropy": 1.25}}


@pytest.fixture
def mock_wepr_weights():
    return {
        "intercept": 0.1,
        "coefficients": {
            "mean_rank_1": 0.5,
            "mean_rank_2": 0.3,
            "max_rank_1": -0.2,
            "max_rank_2": -0.1,
        },
    }


# Tests for EPR Configuration


# We point the patch to where load_weights is imported and used inside PretrainedLogisticRegression
@patch("artefactual.scoring.pretrained_regression.load_weights")
def test_from_pretrained_epr(mock_load, mock_epr_weights):
    mock_load.return_value = mock_epr_weights

    clf = PretrainedLogisticRegression.from_pretrained("dummy_path.json")

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


@patch("artefactual.scoring.pretrained_regression.load_weights")
def test_from_pretrained_wepr(mock_load, mock_wepr_weights):
    mock_load.return_value = mock_wepr_weights

    clf = PretrainedLogisticRegression.from_pretrained("dummy_path.json")

    # Assert correct parameters parsed (k=2, so 2*k = 4 features)
    assert clf.n_features_in_ == 4

    # Expected order: [mean_rank_1, mean_rank_2, max_rank_1, max_rank_2]
    expected_coef = np.array([[0.5, 0.3, -0.2, -0.1]], dtype=np.float32)
    np.testing.assert_array_almost_equal(clf.coef_, expected_coef)
    np.testing.assert_array_almost_equal(clf.intercept_, np.array([0.1], dtype=np.float32))


# Scikit-Learn Compatibility Tests


@patch("artefactual.scoring.pretrained_regression.load_weights")
def test_sklearn_methods_work_without_fit(mock_load, mock_epr_weights):
    mock_load.return_value = mock_epr_weights
    clf = PretrainedLogisticRegression.from_pretrained("dummy_path.json")

    # Using dummy input data matching n_features_in_ (shape: n_samples, n_features)
    x = np.array([[0.8], [0.2]], dtype=np.float32)

    preds = clf.predict(x)
    probs = clf.predict_proba(x)

    assert preds.shape == (2,)
    assert probs.shape == (2, 2)
    np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(2))
