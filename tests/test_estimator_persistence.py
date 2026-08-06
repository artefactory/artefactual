"""Persisting and reading estimators.

Every test names a path. `resolve_estimator` returns a local file without consulting the
Hub, so the suite exercises the whole loading path without a network, credentials, or a
stand-in for either. What is deliberately *not* covered here is the download itself: it is
the one step that needs the Hub, and it is reached only after the pure decisions -- which
file, which repository -- have been made, each of which is tested below on its own.
"""

import numpy as np
import pytest
from conftest import estimators, fitted_logistic, write_estimator
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from artefactual.scoring import epr, wepr
from artefactual.scoring.base_detector import BaseDetector


class AlwaysSure(ClassifierMixin, BaseEstimator):
    """A caller-supplied classifier, of the kind `classifier=` accepts.

    Defined here rather than imported so it is a type skops has no reason to trust, which
    is the condition the refusal path exists for.
    """

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.n_features_in_ = np.shape(x)[1]
        return self

    def predict_proba(self, x):
        return np.tile([0.0, 1.0], (len(x), 1))


# --- reading ---------------------------------------------------------------------------


@given(detector=estimators())
@settings(deadline=None, max_examples=25)
def test_a_detector_survives_a_round_trip(tmp_path_factory, detector):
    path = write_estimator(tmp_path_factory.mktemp("round-trip"), "model.skops", detector)

    restored = BaseDetector.read_estimator(path)

    x = np.linspace(-3, 3, 7 * detector.n_features_in_).reshape(-1, detector.n_features_in_)
    assert np.array_equal(restored.predict_proba(x), detector.predict_proba(x))
    assert restored.coef_.dtype == np.float64


def test_a_sklearn_detector_needs_nothing_trusted(tmp_path):
    # every scikit-learn estimator is trusted by skops itself, which is why the published
    # estimators -- a plain LogisticRegression -- load with no trusted list at all
    forest = RandomForestClassifier(n_estimators=2).fit(np.zeros((4, 2)), [0, 1, 0, 1])
    path = write_estimator(tmp_path, "model.skops", forest)

    assert BaseDetector.read_estimator(path).n_estimators == 2


def test_a_detector_holding_an_unasked_for_type_is_refused(tmp_path):
    # a caller's own estimator, which is what `classifier=` accepts -- not a corrupt file
    path = write_estimator(tmp_path, "model.skops", AlwaysSure().fit(np.zeros((2, 1)), [0, 1]))

    with pytest.raises(ValueError, match="does not load by default"):
        BaseDetector.read_estimator(path)


def test_naming_the_type_makes_it_readable(tmp_path):
    path = write_estimator(tmp_path, "model.skops", AlwaysSure().fit(np.zeros((2, 1)), [0, 1]))

    with pytest.raises(ValueError) as refusal:
        BaseDetector.read_estimator(path)
    # the refusal names the type to pass, so reading the message is the whole fix
    assert AlwaysSure.__name__ in str(refusal.value)

    restored = BaseDetector.read_estimator(path, trusted=[f"{AlwaysSure.__module__}.{AlwaysSure.__qualname__}"])
    assert isinstance(restored, AlwaysSure)


# --- resolving -------------------------------------------------------------------------


def test_a_file_resolves_to_itself(tmp_path):
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    assert BaseDetector.resolve_estimator(path) == path


def test_a_directory_resolves_to_the_model_inside_it(tmp_path):
    # the layout a downloaded repository has, so a clone loads the same way a name does
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    assert BaseDetector.resolve_estimator(tmp_path) == path


# --- the estimators ---------------------------------------------------------------------


@given(k=st.integers(2, 8), data=st.data())
@settings(deadline=None, max_examples=15)
def test_wepr_scores_with_the_model_it_was_given(tmp_path_factory, k, data):
    published = data.draw(estimators(n_features=2 * k))
    path = write_estimator(tmp_path_factory.mktemp("wepr"), "model.skops", published)

    built = wepr(str(path), k=k)

    assert built.estimator.n_features_in_ == 2 * k
    assert np.array_equal(built.estimator.coef_, published.coef_)


def test_a_detector_fit_at_another_rank_count_is_refused(tmp_path):
    detector = fitted_logistic(0.0, [1.0] * 20)  # 2k features, so k=10
    path = write_estimator(tmp_path, "model.skops", detector)

    with pytest.raises(ValueError, match="pass k=10"):
        wepr(str(path), k=15)


def test_an_epr_detector_in_a_wepr_detector_is_refused(tmp_path):
    # the single pooled-entropy coefficient cannot feed a 2k-feature reduction
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    with pytest.raises(ValueError, match="takes 1 feature"):
        wepr(str(path), k=15)


def test_a_detector_saves_the_calibration_it_scores_with(tmp_path):
    detector = fitted_logistic(-0.5, [2.0])
    source = write_estimator(tmp_path, "model.skops", detector)
    detector = epr(str(source), k=15)

    destination = tmp_path / "saved"
    destination.mkdir()
    written = detector.save_estimator(destination)

    reloaded = epr(str(written), k=15)
    x = np.linspace(-2, 2, 5).reshape(-1, 1)
    assert np.array_equal(
        reloaded.estimator.predict_proba(x),
        detector.estimator.predict_proba(x),
    )


def test_saving_an_unfitted_detector_is_refused(tmp_path):
    """A file written before `fit` looks valid and fails only when something scores with it.

    The failure then surfaces from inside the pipeline, at load time, as a missing
    attribute -- nowhere near the call that forgot to fit. Refusing here names it instead.
    """
    with pytest.raises(NotFittedError):
        BaseDetector.trainable("wepr", k=15).save_estimator(tmp_path / "unfitted.skops")

    assert not (tmp_path / "unfitted.skops").exists()


def test_saving_creates_the_parent_directory(tmp_path):
    # the tutorial writes into an output directory the caller has not necessarily made yet
    detector = epr(str(write_estimator(tmp_path, "model.skops", fitted_logistic(-0.5, [2.0]))), k=15)

    written = detector.save_estimator(tmp_path / "new" / "nested" / "model.skops")

    assert written.is_file()
