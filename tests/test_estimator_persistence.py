"""Persisting and reading estimators.

Every test names a path. `PretrainedMixin.cached_file` returns a local file without consulting the
Hub, so the suite exercises the whole loading path without a network, credentials, or a
stand-in for either. What is deliberately *not* covered here is the download itself: it is
the one step that needs the Hub, and it is reached only after the pure decisions -- which
file, which repository -- have been made, each of which is tested below on its own.
"""

import numpy as np
import pytest
import skops.io as sio
from beartype.roar import BeartypeCallHintParamViolation
from conftest import cached_repo, estimators, fitted_logistic, write_estimator
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from artefactual.scoring import EPR, WEPR, BaseDetector
from artefactual.utils.io import MODEL_FILENAME, PretrainedMixin, SavePretrainedMixin


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

    restored = BaseDetector.load_file(path)

    x = np.linspace(-3, 3, 7 * detector.n_features_in_).reshape(-1, detector.n_features_in_)
    assert np.array_equal(restored.predict_proba(x), detector.predict_proba(x))
    assert restored.coef_.dtype == np.float64


def test_a_published_detector_needs_nothing_trusted(tmp_path):
    """A plain `LogisticRegression` loads bare -- the shape every published detector has.

    This is the property `from_pretrained` depends on, and the narrowest one that says so.
    It used to be asserted through a `RandomForestClassifier`, on the premise that skops
    trusts every scikit-learn type; skops 0.15 retired that premise by dropping
    `sklearn.tree._tree.Tree` from its defaults, and the test failed for a reason that had
    nothing to do with this package.
    """
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(-2.91, [58.17]))

    assert BaseDetector.load_file(path).classes_.tolist() == [0, 1]


def test_the_refusal_follows_skops_own_default_list(tmp_path):
    """Whatever skops declines to trust is exactly what `load_file` demands be named.

    Asserted against `get_untrusted_types` rather than against a fixed type name, because
    that list is skops' to change and has changed: a `RandomForestClassifier` needed
    nothing before 0.15 and names `sklearn.tree._tree.Tree` after it. The contract here --
    refuse what skops does not trust, accept it once the caller asks for it by name -- is
    what has to hold across either.
    """
    forest = RandomForestClassifier(n_estimators=2).fit(np.zeros((4, 2)), [0, 1, 0, 1])
    path = write_estimator(tmp_path, "model.skops", forest)

    untrusted = sio.get_untrusted_types(file=path)
    if untrusted:
        with pytest.raises(ValueError, match="does not load by default"):
            BaseDetector.load_file(path)

    assert BaseDetector.load_file(path, trusted=untrusted).n_estimators == 2


def test_a_detector_holding_an_unasked_for_type_is_refused(tmp_path):
    # a caller's own estimator, which is what `classifier=` accepts -- not a corrupt file
    path = write_estimator(tmp_path, "model.skops", AlwaysSure().fit(np.zeros((2, 1)), [0, 1]))

    with pytest.raises(ValueError, match="does not load by default"):
        BaseDetector.load_file(path)


def test_naming_the_type_makes_it_readable(tmp_path):
    path = write_estimator(tmp_path, "model.skops", AlwaysSure().fit(np.zeros((2, 1)), [0, 1]))

    with pytest.raises(ValueError) as refusal:
        BaseDetector.load_file(path)
    # the refusal names the type to pass, so reading the message is the whole fix
    assert AlwaysSure.__name__ in str(refusal.value)

    restored = BaseDetector.load_file(path, trusted=[f"{AlwaysSure.__module__}.{AlwaysSure.__qualname__}"])
    assert isinstance(restored, AlwaysSure)


# --- locating -------------------------------------------------------------------------


def test_a_local_file_is_returned_as_it_is(tmp_path):
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    assert BaseDetector.cached_file(path) == path


def test_a_directory_resolves_to_the_model_inside_it(tmp_path):
    # the layout a downloaded repository has, so a clone loads the same way a name does
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    assert BaseDetector.cached_file(tmp_path) == path
    assert path.name == MODEL_FILENAME


# --- the Hub options, offline (the whole session is cache-only; see conftest) ---


def test_cache_dir_and_revision_reach_the_download(tmp_path):
    commit = "a" * 40
    expected = cached_repo(tmp_path, "org/detector", "release", commit)

    by_branch = BaseDetector.cached_file("org/detector", revision="release", cache_dir=tmp_path, token="unused")
    by_commit = BaseDetector.cached_file("org/detector", revision=commit, cache_dir=tmp_path)

    assert by_branch == by_commit == expected


def test_from_pretrained_forwards_the_hub_options(tmp_path):
    cached_repo(tmp_path, "org/detector", "release", "b" * 40)

    restored = EPR.from_pretrained("org/detector", revision="release", cache_dir=tmp_path, k=15)

    assert restored.estimator.coef_.shape == (1, 1)


def test_an_unknown_revision_is_a_value_error_naming_the_repository(tmp_path):
    cached_repo(tmp_path, "org/detector", "release", "c" * 40)

    with pytest.raises(ValueError, match="org/detector"):
        BaseDetector.cached_file("org/detector", revision="missing", cache_dir=tmp_path)


def test_a_repository_that_is_not_cached_is_a_value_error_offline(tmp_path):
    with pytest.raises(ValueError, match="Could not load an estimator"):
        BaseDetector.cached_file("org/never-published", cache_dir=tmp_path)


def test_a_malformed_repository_id_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="Could not load an estimator"):
        BaseDetector.cached_file("not/a/valid/repo/id", cache_dir=tmp_path)


# --- the estimators ---------------------------------------------------------------------


@given(k=st.integers(2, 8), data=st.data())
@settings(deadline=None, max_examples=15)
def test_wepr_scores_with_the_model_it_was_given(tmp_path_factory, k, data):
    published = data.draw(estimators(n_features=2 * k))
    path = write_estimator(tmp_path_factory.mktemp("wepr"), "model.skops", published)

    built = WEPR.from_pretrained(str(path), k=k)

    assert built.estimator.n_features_in_ == 2 * k
    assert np.array_equal(built.estimator.coef_, published.coef_)


def test_a_detector_fit_at_another_rank_count_is_refused(tmp_path):
    detector = fitted_logistic(0.0, [1.0] * 20)  # 2k features, so k=10
    path = write_estimator(tmp_path, "model.skops", detector)

    with pytest.raises(ValueError, match=r"takes 20 feature\(s\), but k=15 gives 30"):
        WEPR.from_pretrained(str(path), k=15)


def test_an_epr_detector_in_a_wepr_detector_is_refused(tmp_path):
    # the single pooled-entropy coefficient cannot feed a 2k-feature reduction
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    with pytest.raises(ValueError, match="takes 1 feature"):
        WEPR.from_pretrained(str(path), k=15)


def test_a_detector_saves_the_calibration_it_scores_with(tmp_path):
    detector = fitted_logistic(-0.5, [2.0])
    source = write_estimator(tmp_path, "model.skops", detector)
    detector = EPR.from_pretrained(str(source), k=15)

    destination = tmp_path / "saved"
    destination.mkdir()
    written = detector.save_pretrained(destination)

    reloaded = EPR.from_pretrained(str(written), k=15)
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
        WEPR(k=15).save_pretrained(tmp_path / "unfitted.skops")

    assert not (tmp_path / "unfitted.skops").exists()


def test_saving_creates_the_parent_directory(tmp_path):
    # the tutorial writes into an output directory the caller has not necessarily made yet
    detector = EPR.from_pretrained(str(write_estimator(tmp_path, "model.skops", fitted_logistic(-0.5, [2.0]))), k=15)

    written = detector.save_pretrained(tmp_path / "new" / "nested" / "model.skops")

    assert written.is_file()


# --- building an owner from published weights -------------------------------------------


class Owner(PretrainedMixin, SavePretrainedMixin):
    """A minimal owner of an estimator, of the kind the mixin is written for.

    Deliberately not a detector: the mixin resolves, reads and hands over a file, and the
    tests below pin that it does so without knowing what the estimator is for.
    """

    def __init__(self, estimator, label="unlabelled"):
        self._estimator = estimator
        self.label = label

    @property
    def estimator(self):
        return self._estimator

    @classmethod
    def from_estimator(cls, estimator, _identifier, **kwargs):
        return cls(estimator, **kwargs)


class PickyOwner(Owner):
    """An owner that rejects an estimator whose width it cannot score with."""

    @classmethod
    def from_estimator(cls, estimator, identifier, *, n_features, **kwargs):
        if estimator.n_features_in_ != n_features:
            msg = f"'{identifier}' takes {estimator.n_features_in_} feature(s), not {n_features}."
            raise ValueError(msg)
        return cls(estimator, **kwargs)


def test_an_owner_is_built_from_the_file_it_names(tmp_path):
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    owner = Owner.from_pretrained(path)

    assert owner.estimator.n_features_in_ == 1
    assert owner.label == "unlabelled"


def test_keywords_reach_the_owner_rather_than_the_reader(tmp_path):
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    assert Owner.from_pretrained(path, label="named").label == "named"


def test_an_owner_may_refuse_the_estimator_it_is_handed(tmp_path):
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0, 2.0]))

    with pytest.raises(ValueError, match="takes 2 feature"):
        PickyOwner.from_pretrained(path, n_features=1)

    assert PickyOwner.from_pretrained(path, n_features=2).estimator.n_features_in_ == 2


def test_a_class_without_from_estimator_is_rejected(tmp_path):
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    class Bare(PretrainedMixin):
        pass

    with pytest.raises(BeartypeCallHintParamViolation, match="EstimatorFactory"):
        Bare.from_pretrained(path)


def test_an_object_without_an_estimator_is_rejected_by_save_pretrained(tmp_path):
    class Bare(SavePretrainedMixin):
        pass

    with pytest.raises(BeartypeCallHintParamViolation, match="EstimatorOwner"):
        Bare().save_pretrained(tmp_path / "model.skops")

    assert not (tmp_path / "model.skops").exists()


def test_an_untrusted_type_is_refused_before_the_owner_sees_it(tmp_path):
    path = write_estimator(tmp_path, "model.skops", AlwaysSure().fit(np.zeros((2, 1)), [0, 1]))

    with pytest.raises(ValueError, match="AlwaysSure"):
        Owner.from_pretrained(path)

    assert Owner.from_pretrained(path, trusted=["test_estimator_persistence.AlwaysSure"])


# --- the load path, through from_pretrained ---------------------------------------------


@given(published=estimators(), data=st.data())
@settings(deadline=None, max_examples=15)
def test_from_pretrained_hands_the_owner_the_file_as_written(tmp_path_factory, published, data):
    # `from_pretrained` reaches the file by name, by string or by directory
    directory = tmp_path_factory.mktemp("load")
    path = write_estimator(directory, "model.skops", published)
    named = data.draw(st.sampled_from([path, directory, str(path)]))

    owner = Owner.from_pretrained(named)

    x = np.linspace(-3, 3, 5 * published.n_features_in_).reshape(-1, published.n_features_in_)
    assert np.array_equal(owner.estimator.predict_proba(x), published.predict_proba(x))


def test_a_local_path_is_loaded_without_the_hub_options(tmp_path):
    path = write_estimator(tmp_path, "model.skops", fitted_logistic(0.0, [1.0]))

    owner = Owner.from_pretrained(path, revision="ignored", token="ignored", cache_dir=tmp_path / "unused")

    assert owner.estimator.n_features_in_ == 1


def test_save_pretrained_is_usable_without_from_pretrained(tmp_path):
    class SaveOnly(SavePretrainedMixin):
        estimator = fitted_logistic(0.0, [1.0])

    written = SaveOnly().save_pretrained(tmp_path)

    assert written == tmp_path / "model.skops"
    assert Owner.from_pretrained(written).estimator.n_features_in_ == 1


# --- save_pretrained -> from_pretrained -------------------------------------------------


@given(published=estimators(), as_directory=st.booleans())
@settings(deadline=None, max_examples=15)
def test_what_is_saved_is_what_from_pretrained_loads(tmp_path_factory, published, as_directory):
    directory = tmp_path_factory.mktemp("saved")
    target = directory / "out" if as_directory else directory / "named.skops"
    if as_directory:
        target.mkdir()

    written = Owner(published).save_pretrained(target)
    reloaded = Owner.from_pretrained(target)

    assert written == (target / MODEL_FILENAME if as_directory else target)
    x = np.linspace(-3, 3, 5 * published.n_features_in_).reshape(-1, published.n_features_in_)
    assert np.array_equal(reloaded.estimator.predict_proba(x), published.predict_proba(x))
