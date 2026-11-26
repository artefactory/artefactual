"""What the detectors and classifiers keep promising as `Pipeline`s.

The classifiers take logprob arrays and are exercised on random tables and cubes; the
detectors take responses and are exercised on the checked-in Responses payload, because
`check_estimator` only ever sends arrays and would not notice the response path breaking.
"""

import dataclasses
import json
import pickle
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from beartype.door import is_bearable
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from artefactual.preprocessing import LogProbParser
from artefactual.scoring import EPR, WEPR, Pooling
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.entropy_methods.pooling import EPRPooling, WEPRPooling

EXAMPLES = Path(__file__).parent.parent / "docs" / "examples"
RESPONSES = json.loads((EXAMPLES / "open_ai_responses_top15.json").read_text(encoding="utf-8"))["responses"]
LABELS = np.array([0, 1])
CLASSIFIERS = [EPR, WEPR]
ESTIMATORS = CLASSIFIERS
CLASSIFIER_IDS = ["EPR", "WEPR"]
ESTIMATOR_IDS = CLASSIFIER_IDS


def detector(cls):
    """The unfitted responses -> probabilities pipeline over `cls`."""
    return make_pipeline(LogProbParser(k=15), cls())


def table(n=40, width=6, seed=0):
    """Negative logprob rows, with a label that depends on the first column."""
    rng = np.random.RandomState(seed)
    x = -rng.exponential(1.0, size=(n, width))
    y = (x[:, 0] > np.median(x[:, 0])).astype(int)
    return x, y


def fit_data(width=6):
    """Data `cls` fits on: a random table."""
    return table(width=width)


# --- get_params / clone ------------------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_a_default_detector_reports_no_estimator_and_clones(cls):
    detector = cls()
    assert detector.get_params()["estimator"] is None
    twin = clone(detector)
    assert twin.get_params(deep=False).keys() == detector.get_params(deep=False).keys()
    assert twin.estimator is None


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_a_custom_estimator_is_reported_and_cloned(cls):
    custom = LogisticRegression(C=0.5)
    detector = cls(estimator=custom)
    assert detector.get_params()["estimator"] is custom
    twin = clone(detector)
    assert twin.estimator.C == pytest.approx(0.5)
    assert twin.estimator is not custom


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_a_fitted_detector_clones_unfitted(cls):
    detector = cls().fit(*fit_data())
    twin = clone(detector)
    assert "pipeline_" not in vars(twin)
    assert "n_features_in_" not in vars(twin)


# --- search over the real class ----------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_grid_search_runs_over_numeric_data(cls):
    x, y = table()
    grid = {"estimator__C": [0.1, 1.0]}
    search = GridSearchCV(cls(estimator=LogisticRegression()), grid, cv=3).fit(x, y)
    assert set(search.best_params_) == set(grid)
    assert search.best_estimator_.pipeline_[-1].C == search.best_params_["estimator__C"]
    assert search.best_estimator_.predict_proba(x).shape == (len(x), 2)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_from_pretrained_scores_like_the_detector_it_was_saved_from(cls, tmp_path):
    x, y = table(width=5)
    path = cls(estimator=LogisticRegression(C=0.5)).fit(x, y).save_pretrained(tmp_path / "model.skops")
    loaded = cls.from_pretrained(path, **({"k": 5} if cls is EPR else {}))
    assert loaded.get_params()["estimator"] is loaded.estimator
    np.testing.assert_allclose(
        loaded.predict_proba(x), cls(estimator=LogisticRegression(C=0.5)).fit(x, y).predict_proba(x)
    )


# --- targets -----------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
@pytest.mark.parametrize(
    "y",
    [np.zeros(40), np.arange(40) % 3, np.linspace(0, 1, 40), np.r_[np.nan, np.zeros(39)]],
    ids=["one-class", "multiclass", "continuous", "nan"],
)
def test_non_binary_or_invalid_targets_are_rejected(cls, y):
    with pytest.raises(ValueError, match=r"binary|NaN|continuous|Unknown label|class"):
        cls().fit(table()[0], y)


# --- response path, by hand --------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_response_input_fits_scores_and_pickles(cls):
    pipeline = detector(cls).fit(RESPONSES, LABELS)
    scores = pipeline.predict_proba(RESPONSES)
    assert scores.shape == (2, 2)
    restored = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(restored.predict_proba(RESPONSES), scores)


def responses_as(container):
    """The two checked-in responses held by a 1-D array or Series rather than a list."""
    if container == "object array":
        column = np.empty(len(RESPONSES), dtype=object)
        column[:] = RESPONSES
        return column
    return pd.Series(RESPONSES)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
@pytest.mark.parametrize("container", ["object array", "series"])
def test_a_one_dimensional_container_of_responses_is_read_as_responses(cls, container):
    held = responses_as(container)
    pipeline = detector(cls).fit(held, LABELS)
    from_list = detector(cls).fit(RESPONSES, LABELS)
    np.testing.assert_array_equal(pipeline.predict_proba(held), from_list.predict_proba(RESPONSES))
    np.testing.assert_array_equal(
        pipeline[-1].predict_token_proba(pipeline[0].transform(held)),
        from_list[-1].predict_token_proba(from_list[0].transform(RESPONSES)),
    )


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_a_two_dimensional_object_table_of_numbers_is_still_a_table(cls):
    x, y = table()
    from_object = cls().fit(x.astype(object), y).predict_proba(x.astype(object))
    np.testing.assert_allclose(from_object, cls().fit(x, y).predict_proba(x))


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_predict_leaves_the_response_untouched(cls):
    pipeline = detector(cls).fit(RESPONSES, LABELS)
    before = deepcopy(RESPONSES)
    pipeline.predict_proba(RESPONSES)
    pipeline.predict(RESPONSES)
    pipeline[-1].predict_token_proba(pipeline[0].transform(RESPONSES))
    assert RESPONSES == before


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_fit_leaves_the_response_untouched(cls):
    before = deepcopy(RESPONSES)
    detector(cls).fit(RESPONSES, LABELS)
    assert RESPONSES == before


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_repeated_fit_and_predict_give_identical_results(cls):
    pipeline = detector(cls).fit(RESPONSES, LABELS)
    first = pipeline.predict_proba(RESPONSES)
    np.testing.assert_array_equal(pipeline.predict_proba(RESPONSES), first)
    refitted = pipeline.fit(RESPONSES, LABELS)
    np.testing.assert_array_equal(refitted.predict_proba(RESPONSES), first)
    np.testing.assert_array_equal(detector(cls).fit(RESPONSES, LABELS).predict_proba(RESPONSES), first)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_a_clone_and_the_split_pipeline_score_like_the_detector(cls):
    pipeline = detector(cls).fit(RESPONSES, LABELS)
    expected = pipeline.predict_proba(RESPONSES)
    features = pipeline[0].transform(RESPONSES)
    np.testing.assert_allclose(pipeline[-1].predict_proba(features), expected)
    np.testing.assert_allclose(clone(pipeline).fit(RESPONSES, LABELS).predict_proba(RESPONSES), expected)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_construction_sets_only_the_declared_parameters(cls):
    detector = cls()
    assert set(vars(detector)) == set(detector.get_params())


# --- set_params round trips --------------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
@pytest.mark.parametrize("deep", [False, True])
def test_set_params_of_get_params_keeps_a_working_classifier(cls, deep):
    detector = cls()
    detector.set_params(**detector.get_params(deep=deep))
    x, y = fit_data()
    assert detector.fit(x, y).predict_proba(x).shape == (len(y), 2)
    assert type(detector.pipeline_[-1]) is LogisticRegression


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_setting_the_estimator_to_none_restores_the_default(cls):
    detector = cls(estimator=LogisticRegression(C=0.5)).set_params(estimator=None).fit(*fit_data())
    assert detector.pipeline_[-1].C == pytest.approx(np.inf)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_grid_search_over_the_estimator_including_the_default(cls):
    x, y = table()
    grid = {"estimator": [None, LogisticRegression(C=1.0)]}
    search = GridSearchCV(cls(), grid, cv=2).fit(x, y)
    assert len(search.cv_results_["params"]) == 2
    assert not np.isnan(search.cv_results_["mean_test_score"]).any()


# --- the reduction is fixed by the class --------------------------------------------------


# --- feature names and set_output -------------------------------------------------------


K = 15


def fitted(cls):
    """`cls` fitted at `K` ranks on a random table."""
    return cls().fit(*fit_data(width=K))


def front(cls):
    """The fitted features step of `cls`, fitted at `K` ranks."""
    return fitted(cls).pipeline_[:-1]


def front_input():
    """What the steps before the classifier take: a table."""
    return table(width=K)[0]


def entropy_input():
    """What the entropy step takes: the table itself."""
    return table(width=K)[0]


def entropy_slice(estimator):
    """The entropy step alone, as a one-step `Pipeline`."""
    return estimator[-2:-1]


@pytest.mark.parametrize("cls", [EPR], ids=["EPR"])
def test_epr_names_its_one_feature(cls):
    assert list(front(cls).get_feature_names_out()) == ["epr"]


@pytest.mark.parametrize("cls", [WEPR], ids=["WEPR"])
def test_wepr_names_mean_then_max_per_rank(cls):
    names = list(front(cls).get_feature_names_out())
    assert names == [f"wepr_{b}_rank{i}" for b in ("mean", "max") for i in range(K)]


@pytest.mark.parametrize("pooling", [EPRPooling(), WEPRPooling()], ids=["EPR", "WEPR"])
def test_the_names_of_a_narrow_table_match_the_columns_it_produces(pooling):
    """Names follow the fitted width, not `k`, for a table narrower than `k`."""
    step = EntropyTransformer(pooling=pooling).fit(table(n=6, width=3)[0])
    features = step.transform(table(n=6, width=3)[0])
    assert len(step.get_feature_names_out()) == features.shape[1]
    assert len(step.set_output(transform="pandas").transform(table(n=6, width=3)[0]).columns) == features.shape[1]


def test_the_names_of_a_narrow_wepr_table_are_those_of_its_own_ranks():
    step = EntropyTransformer(pooling=WEPRPooling()).fit(table(n=6, width=3)[0])
    expected = [f"wepr_{b}_rank{i}" for b in ("mean", "max") for i in range(3)]
    assert list(step.get_feature_names_out()) == expected


def test_a_refit_on_a_cube_names_the_cube_width_not_the_stale_table_width():
    step = EntropyTransformer(pooling=WEPRPooling()).fit(table(n=6, width=3)[0])
    assert len(step.get_feature_names_out()) == 6
    step.fit(-np.ones((2, 4, K)))
    assert len(step.get_feature_names_out()) == 2 * K


def test_a_refit_on_a_cube_forgets_the_column_names_of_the_table():
    frame = pd.DataFrame(table(n=6, width=3)[0], columns=list("abc"))
    step = EntropyTransformer(pooling=WEPRPooling()).fit(frame).fit(-np.ones((2, 4, 3)))
    assert not hasattr(step, "feature_names_in_")
    assert list(step.get_feature_names_out()) == [f"wepr_{b}_rank{i}" for b in ("mean", "max") for i in range(3)]


def test_input_features_are_checked_against_the_fitted_width_when_narrower_than_k():
    step = EntropyTransformer(pooling=WEPRPooling()).fit(table(n=6, width=3)[0])
    assert step.get_feature_names_out(["a", "b", "c"]).tolist() == [
        f"wepr_{b}_{r}" for b in ("mean", "max") for r in "abc"
    ]
    with pytest.raises(ValueError, match="input_features"):
        step.get_feature_names_out([f"r{i}" for i in range(K)])


def test_a_narrow_named_frame_names_the_columns_it_produces():
    frame = pd.DataFrame(table(n=6, width=3)[0], columns=list("abc"))
    step = EntropyTransformer(pooling=WEPRPooling()).fit(frame).set_output(transform="pandas")
    assert list(step.transform(frame).columns) == list(step.get_feature_names_out())


@pytest.mark.parametrize(("fitted", "given"), [(3, K), (K, 3)], ids=["wider", "narrower"])
def test_a_cube_of_another_width_than_the_fit_is_refused_at_transform(fitted, given):
    step = EntropyTransformer(pooling=WEPRPooling()).fit(-np.ones((2, 4, fitted)))
    with pytest.raises(ValueError, match=rf"X has {given} features, but EntropyTransformer is expecting {fitted}"):
        step.transform(-np.ones((2, 4, given)))


def test_an_empty_cube_is_not_refused_by_a_fitted_width():
    step = EntropyTransformer(pooling=WEPRPooling()).fit(-np.ones((2, 4, K)))
    assert step.validate(np.empty((0, 0, 0))).shape == (0, 0, 0)


def test_a_cube_fit_with_no_k_names_every_rank_it_was_fitted_on():
    step = EntropyTransformer(pooling=WEPRPooling()).fit(-np.ones((2, 4, 3)))
    assert len(step.get_feature_names_out()) == 6


def test_the_names_agree_with_the_coefficients_a_classifier_of_that_k_holds():
    """A pretrained file holds an unnamed `LogisticRegression`, so names are checked by count and order.

    Columns reach the classifier in the order of the names, so a coefficient position
    means the rank-wise feature named at that position.
    """
    x, y = table(width=5)
    classifier = WEPR().fit(x, y)
    names = classifier.pipeline_[:-1].get_feature_names_out()
    assert len(names) == classifier.pipeline_[-1].coef_.shape[1]
    features = classifier.pipeline_[:-1].set_output(transform="pandas").transform(x)
    assert list(features.columns) == list(names)


@pytest.mark.parametrize("cls", [WEPR], ids=["WEPR"])
def test_wepr_names_follow_the_column_order_of_the_features(cls):
    """The mean branch comes first: per rank, a mean never exceeds the max it is named beside."""
    pd = pytest.importorskip("pandas")
    estimator = fitted(cls)
    features = estimator.pipeline_[:-1].set_output(transform="pandas").transform(front_input())
    assert is_bearable(features, pd.DataFrame)
    means = features.filter(like="wepr_mean_").to_numpy()
    maxes = features.filter(like="wepr_max_").to_numpy()
    assert list(features.columns) == list(estimator.pipeline_[:-1].get_feature_names_out())
    np.testing.assert_array_equal(features.to_numpy(), np.hstack([means, maxes]))
    assert np.all(means <= maxes + 1e-12)
    assert features.shape[1] == 2 * K


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_the_front_of_an_estimator_returns_a_named_frame(cls, backend):
    pytest.importorskip(backend)
    names = list(front(cls).get_feature_names_out())
    out = front(cls).set_output(transform=backend).transform(front_input())
    assert type(out).__module__.startswith(backend)
    assert list(out.columns) == names


@pytest.mark.parametrize("cls", [WEPR], ids=["WEPR"])
def test_transform_tokens_stays_three_dimensional_under_set_output(cls):
    pytest.importorskip("pandas")
    entropy = fitted(cls).pipeline_[-2].set_output(transform="pandas")
    assert entropy.transform_tokens(entropy_input()).ndim == 3


def test_the_parser_is_not_wrapped_into_a_frame():
    pytest.importorskip("pandas")
    cube = detector(WEPR)[:1].set_output(transform="pandas").transform(RESPONSES)
    assert cube.ndim == 3


def test_a_custom_pooling_names_its_features_after_itself():
    def mean_of_ranks(s, axis):
        return np.nanmean(np.nanmean(s, axis=-1, keepdims=True), axis=axis)

    step = EntropyTransformer(pooling=mean_of_ranks).fit(front_input())
    assert list(step.get_feature_names_out()) == ["mean_of_ranks"]


@pytest.mark.parametrize("cls", [EPR], ids=["EPR"])
def test_set_output_rejects_unknown_formats_when_the_output_is_built(cls):
    step = front(cls).set_output(transform="numpy")
    with pytest.raises(ValueError, match="numpy"):
        step.transform(front_input())


def test_a_classifier_fitted_on_a_frame_names_its_features_after_the_columns():
    columns = ["a", "b", "c"]
    x, y = table(width=3)
    classifier = WEPR().fit(pd.DataFrame(x, columns=columns), y)
    assert list(classifier.pipeline_[:-1].get_feature_names_out()) == [
        f"wepr_{branch}_{name}" for branch in ("mean", "max") for name in columns
    ]
    with pytest.raises(ValueError, match="feature_names_in_"):
        classifier.pipeline_[:-1].get_feature_names_out(["x", "y", "z"])


def test_a_classifier_fitted_on_an_array_names_its_features_after_the_ranks():
    x, y = table(width=3)
    classifier = WEPR().fit(x, y)
    assert list(classifier.pipeline_[:-1].get_feature_names_out()) == [
        f"wepr_{branch}_rank{i}" for branch in ("mean", "max") for i in range(3)
    ]


# --- k reaches every step ----------------------------------------------------------------


# --- a classifier takes arrays and the parser takes responses ----------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_the_parser_refuses_a_table(cls):
    x, y = table()
    with pytest.raises(TypeError):
        detector(cls).fit(x, y)


# --- package root exports ----------------------------------------------------------------


def test_the_estimators_are_exported_from_the_package_root_and_from_scoring():
    import artefactual
    import artefactual.scoring

    for cls in ESTIMATORS:
        assert vars(artefactual)[cls.__name__] is cls
        assert vars(artefactual.scoring)[cls.__name__] is cls
    assert {cls.__name__ for cls in ESTIMATORS} <= set(artefactual.__all__)


# --- slicing -----------------------------------------------------------------------------


# --- n_features_in_ ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_n_features_in_is_the_width_of_the_fitted_table_and_absent_before(cls):
    detector = cls()
    with pytest.raises(AttributeError):
        detector.n_features_in_  # ruff: ignore[useless-expression]
    x, y = table(width=6)
    assert detector.fit(x, y).n_features_in_ == 6


# --- parameter validation ----------------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_fit_rejects_an_estimator_without_predict_proba(cls):
    x, y = fit_data()
    from sklearn.svm import LinearSVC

    with pytest.raises(ValueError, match="'estimator'"):
        cls(estimator=LinearSVC()).fit(x, y)


# --- labels ------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_any_two_labels_are_accepted_and_the_second_is_column_one(cls):
    x, y = table()
    detector = cls().fit(x, y + 1)
    assert list(detector.classes_) == [1, 2]
    zero_one = cls().fit(x, y)
    assert np.array_equal(detector.predict_proba(x), zero_one.predict_proba(x))


# --- loading a fitted estimator ----------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_from_estimator_rejects_an_unfitted_estimator(cls):
    with pytest.raises(ValueError, match="not fitted"):
        cls.from_estimator(LogisticRegression(), "somewhere.skops")


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_from_estimator_rejects_a_non_classifier(cls):
    from sklearn.linear_model import LinearRegression

    x, y = table()
    with pytest.raises(ValueError, match="not a classifier"):
        cls.from_estimator(LinearRegression().fit(x, y), "somewhere.skops")


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_from_estimator_rejects_more_than_two_classes(cls):
    x, _ = table()
    three = LogisticRegression().fit(x, np.arange(len(x)) % 3)
    with pytest.raises(ValueError, match="3 class"):
        cls.from_estimator(three, "somewhere.skops")


def test_the_user_api_imports_from_the_package_root_and_the_mixin_does_not():
    import artefactual

    names = {
        "EPR", "WEPR", "Judgment", "LogProbParser", "EntropyTransformer",
        "ChatCompletion", "ResponsesPayload", "BatchRequestOutput", "read_batch", "epr_reduction", "wepr_reduction",
    }  # fmt: skip
    assert names <= set(artefactual.__all__)
    assert "LogprobsMixin" not in artefactual.__all__


# --- a pooling swapped after the fit -------------------------------------------------------


def test_a_fitted_step_refuses_a_pooling_swapped_after_the_fit():
    x, _ = table(width=5)
    step = EntropyTransformer(pooling=ScaledMean(2)).fit(x)
    step.set_params(pooling=ScaledMean(3))
    with pytest.raises(ValueError, match="refit"):
        step.transform(x)


def test_a_step_never_fitted_transforms_with_any_pooling():
    x, _ = table(width=5)
    assert EntropyTransformer(pooling=WEPRPooling()).transform(x).shape == (len(x), 10)


def test_a_pooling_cannot_be_mutated():
    pooling = WEPRPooling()
    with pytest.raises(dataclasses.FrozenInstanceError):
        pooling.extra = 2  # ty: ignore[invalid-assignment]


@dataclasses.dataclass(frozen=True)
class ScaledMean(Pooling):
    """One mean per rank, scaled by a number."""

    scale: float

    def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
        return np.nanmean(X * self.scale, axis=axis)

    def mean(self, x, axis):
        return np.nanmean(x, axis=axis)


@dataclasses.dataclass(frozen=True)
class ScaledByArray(Pooling):
    """One mean per rank, scaled by an array that equality cannot compare."""

    scale: Any = dataclasses.field(compare=False)
    constant_width: ClassVar[bool] = False  # `scale` fits one `k`, so the count cannot be probed at others

    def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
        return np.nanmean(X * self.scale, axis=axis)


def rank_percentile(x, axis, q):
    return np.nanpercentile(x, q, axis=axis)


class MeanCallable:
    """A callable object that defines no equality."""

    def __call__(self, x, axis):
        return np.nanmean(x, axis=axis)


@dataclasses.dataclass(frozen=True)
class ScaledBy(Pooling):
    """Holds a callable as an attribute and applies it."""

    fn: Any

    def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
        return self.fn(X, axis=axis)


CUSTOM_POOLINGS = {
    "partial": partial(rank_percentile, q=95),
    "callable-object": MeanCallable(),
    "bound-method": ScaledMean(1.0).mean,
    "array-attribute": ScaledByArray(np.ones(5)),
    "callable-attribute": ScaledBy(partial(np.nanmean)),
}


@pytest.mark.parametrize(
    "copied",
    [lambda x: x, deepcopy, lambda x: pickle.loads(pickle.dumps(x))],
    ids=["itself", "deepcopy", "pickle"],
)
@pytest.mark.parametrize("pooling", CUSTOM_POOLINGS.values(), ids=CUSTOM_POOLINGS.keys())
def test_a_fitted_pooling_is_recognised_as_the_one_it_was_fitted_with(pooling, copied):
    x, _ = table(width=5)
    step = EntropyTransformer(pooling=pooling).fit(x)
    np.testing.assert_array_equal(copied(step).transform(x), step.transform(x))
