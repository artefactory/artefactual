"""What the detectors and classifiers keep promising as `Pipeline`s.

The classifiers take logprob arrays and are exercised on random tables and cubes; the
detectors take responses and are exercised on the checked-in Responses payload, because
`check_estimator` only ever sends arrays and would not notice the response path breaking.
"""

import json
import pickle
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from beartype.door import is_bearable
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from artefactual.scoring import EPR, WEPR, EPRClassifier, Pooling, PoolingDetector, WEPRClassifier
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.entropy_methods.pooling import EPRPooling, WEPRPooling

EXAMPLES = Path(__file__).parent.parent / "docs" / "examples"
RESPONSES = json.loads((EXAMPLES / "open_ai_responses_top15.json").read_text(encoding="utf-8"))["responses"]
LABELS = np.array([0, 1])
DETECTORS = [EPR, WEPR]
CLASSIFIERS = [EPRClassifier, WEPRClassifier]
ESTIMATORS = [*DETECTORS, *CLASSIFIERS]
IDS = ["EPR", "WEPR"]
CLASSIFIER_IDS = ["EPRClassifier", "WEPRClassifier"]
ESTIMATOR_IDS = [*IDS, *CLASSIFIER_IDS]


def table(n=40, width=6, seed=0):
    """Negative logprob rows, with a label that depends on the first column."""
    rng = np.random.RandomState(seed)
    x = -rng.exponential(1.0, size=(n, width))
    y = (x[:, 0] > np.median(x[:, 0])).astype(int)
    return x, y


def fit_data(cls, width=6):
    """Data `cls` fits on: the checked-in responses for a detector, a random table otherwise."""
    return (RESPONSES, LABELS) if cls in DETECTORS else table(width=width)


# --- get_params / clone ------------------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_a_default_detector_reports_no_estimator_and_clones(cls):
    detector = cls(k=15)
    assert detector.get_params()["estimator"] is None
    twin = clone(detector)
    assert twin.get_params(deep=False).keys() == detector.get_params(deep=False).keys()
    assert type(twin.estimator) is LogisticRegression
    assert twin.estimator is not detector.estimator


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_a_custom_estimator_is_reported_and_cloned(cls):
    custom = LogisticRegression(C=0.5)
    detector = cls(estimator=custom, k=15)
    assert detector.get_params()["estimator"] is custom
    twin = clone(detector)
    assert twin.estimator.C == pytest.approx(0.5)
    assert twin.estimator is not custom


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_a_fitted_detector_clones_unfitted(cls):
    detector = cls(k=15).fit(*fit_data(cls))
    twin = clone(detector)
    assert not hasattr(twin.estimator, "coef_")
    assert not hasattr(twin, "n_features_in_")


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_a_default_with_a_changed_nested_param_is_reported_as_itself(cls):
    detector = cls(k=15).set_params(classifier__C=2.0)
    assert detector.get_params()["estimator"] is detector.estimator
    assert clone(detector).estimator.C == pytest.approx(2.0)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
@pytest.mark.parametrize("name", ["estimator__C", "classifier__C"])
def test_a_default_detector_takes_the_classifier_param_under_either_name(cls, name):
    detector = cls(k=15).set_params(**{name: 1.0})
    assert detector.steps[-1][1].C == pytest.approx(1.0)
    assert clone(detector).steps[-1][1].C == pytest.approx(1.0)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_the_pipeline_surface_is_kept(cls):
    detector = cls(k=4)
    assert is_bearable(detector, Pipeline)
    steps = ["parser", "entropy", "classifier"] if cls in DETECTORS else ["entropy", "classifier"]
    assert [name for name, _ in detector.steps] == steps
    assert detector.named_steps["classifier"] is detector.estimator
    assert detector.get_params()["entropy__k"] == 4
    assert detector.get_params()["classifier__C"] == np.inf
    assert type(detector[:-1]) is Pipeline
    assert detector.set_params(entropy__k=3).named_steps["entropy"].k == 3


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_fit_keeps_the_steps_list_identity(cls):
    detector = cls(k=15)
    steps = detector.steps
    detector.fit(*fit_data(cls))
    assert detector.steps is steps


# --- search over the real class ----------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
@pytest.mark.parametrize("grid", [{"k": [2, 4]}, {"classifier__C": [0.1, 1.0]}])
def test_grid_search_runs_over_numeric_data(cls, grid):
    x, y = table()
    search = GridSearchCV(cls(k=15), grid, cv=3).fit(x, y)
    assert set(search.best_params_) == set(grid)
    assert search.best_estimator_.predict_proba(x).shape == (len(x), 2)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_from_pretrained_then_get_params_and_clone(cls, tmp_path):
    x, y = table(width=5)
    path = cls(k=5, estimator=LogisticRegression(C=0.5)).fit(x, y).save_pretrained(tmp_path / "model.skops")
    loaded = cls.from_pretrained(path, k=5)
    assert loaded.get_params()["estimator"] is loaded.estimator
    twin = clone(loaded)
    assert twin.k == 5
    assert twin.estimator.C == pytest.approx(0.5)
    assert not hasattr(twin.estimator, "coef_")  # clone is unfitted, as for any estimator
    np.testing.assert_allclose(
        loaded.predict_proba(x), cls(k=5, estimator=LogisticRegression(C=0.5)).fit(x, y).predict_proba(x)
    )


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_from_pretrained_of_a_default_classifier_clones(cls, tmp_path):
    x, y = table(width=5)
    path = cls(k=5).fit(x, y).save_pretrained(tmp_path / "model.skops")
    loaded = cls.from_pretrained(path, k=5)
    assert loaded.get_params()["estimator"] is None
    assert clone(loaded).k == 5


# --- narrow tables -----------------------------------------------------------------------


def test_a_table_narrower_than_k_round_trips_only_at_its_own_width(tmp_path):
    """A narrow table is accepted, and a file fit on one is loaded at the matching k.

    Rejecting narrow tables at fit would stop `check_estimator`, whose toy data has a
    couple of columns against `k=15`, so the width is reported at load time
    instead, where the error states the width the file takes.
    """
    x, y = table(width=5)
    path = WEPRClassifier(k=15).fit(x, y).save_pretrained(tmp_path / "model.skops")
    with pytest.raises(ValueError, match=r"takes 10 feature\(s\), but k=15 gives 30"):
        WEPRClassifier.from_pretrained(path, k=15)
    assert WEPRClassifier.from_pretrained(path, k=5).predict_proba(x).shape == (len(x), 2)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_a_table_wider_than_k_is_cut_to_k(cls):
    x, y = table(width=9)
    wide = cls(k=4).fit(x, y).predict_proba(x)
    np.testing.assert_allclose(wide, cls(k=4).fit(x[:, :4], y).predict_proba(x[:, :4]))


# --- targets -----------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
@pytest.mark.parametrize(
    "y",
    [np.zeros(40), np.arange(40) % 3, np.linspace(0, 1, 40), np.r_[np.nan, np.zeros(39)]],
    ids=["one-class", "multiclass", "continuous", "nan"],
)
def test_non_binary_or_invalid_targets_are_rejected(cls, y):
    with pytest.raises(ValueError, match=r"binary|NaN|continuous|Unknown label|class"):
        cls(k=15).fit(table()[0], y)


# --- response path, by hand --------------------------------------------------------------


@pytest.mark.parametrize("cls", DETECTORS, ids=IDS)
def test_response_input_still_fits_scores_and_pickles(cls):
    detector = cls(k=15).fit(RESPONSES, LABELS)
    scores = detector.predict_proba(RESPONSES)
    assert scores.shape == (2, 2)
    restored = pickle.loads(pickle.dumps(detector))
    np.testing.assert_array_equal(restored.predict_proba(RESPONSES), scores)


def responses_as(container):
    """The two checked-in responses held by a 1-D array or Series rather than a list."""
    if container == "object array":
        column = np.empty(len(RESPONSES), dtype=object)
        column[:] = RESPONSES
        return column
    return pd.Series(RESPONSES)


@pytest.mark.parametrize("cls", DETECTORS, ids=IDS)
@pytest.mark.parametrize("container", ["object array", "series"])
def test_a_one_dimensional_container_of_responses_is_read_as_responses(cls, container):
    held = responses_as(container)
    detector = cls(k=15).fit(held, LABELS)
    from_list = cls(k=15).fit(RESPONSES, LABELS)
    np.testing.assert_array_equal(detector.predict_proba(held), from_list.predict_proba(RESPONSES))
    for held_scores, list_scores in zip(
        detector.predict_token_proba(held), from_list.predict_token_proba(RESPONSES), strict=True
    ):
        np.testing.assert_array_equal(held_scores, list_scores)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_a_two_dimensional_object_table_of_numbers_is_still_a_table(cls):
    x, y = table()
    from_object = cls(k=15).fit(x.astype(object), y).predict_proba(x.astype(object))
    np.testing.assert_allclose(from_object, cls(k=15).fit(x, y).predict_proba(x))


@pytest.mark.parametrize("cls", DETECTORS, ids=IDS)
def test_predict_leaves_the_response_untouched(cls):
    detector = cls(k=15).fit(RESPONSES, LABELS)
    before = deepcopy(RESPONSES)
    detector.predict_proba(RESPONSES)
    detector.predict(RESPONSES)
    detector.predict_token_proba(RESPONSES)
    assert RESPONSES == before


@pytest.mark.parametrize("cls", DETECTORS, ids=IDS)
def test_fit_leaves_the_response_untouched(cls):
    before = deepcopy(RESPONSES)
    cls(k=15).fit(RESPONSES, LABELS)
    assert RESPONSES == before


@pytest.mark.parametrize("cls", DETECTORS, ids=IDS)
def test_a_clone_and_the_split_pipeline_score_like_the_detector(cls):
    detector = cls(k=15).fit(RESPONSES, LABELS)
    expected = detector.predict_proba(RESPONSES)
    features = detector[:-1].transform(RESPONSES)
    np.testing.assert_allclose(detector.estimator.predict_proba(features), expected)
    np.testing.assert_allclose(clone(detector).fit(RESPONSES, LABELS).predict_proba(RESPONSES), expected)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_construction_sets_only_the_declared_parameters(cls):
    detector = cls(k=15)
    assert set(vars(detector)) == {"steps", "memory", "transform_input", "verbose"}


# --- set_params round trips --------------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
@pytest.mark.parametrize("deep", [False, True])
def test_set_params_of_get_params_keeps_a_working_classifier(cls, deep):
    detector = cls(k=15)
    detector.set_params(**detector.get_params(deep=deep))
    assert type(detector.steps[-1][1]) is LogisticRegression
    x, y = fit_data(cls)
    assert detector.fit(x, y).predict_proba(x).shape == (len(y), 2)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_setting_the_estimator_to_none_restores_the_default(cls):
    detector = cls(estimator=LogisticRegression(C=0.5), k=15).set_params(estimator=None)
    assert detector.estimator.get_params() == LogisticRegression(C=np.inf, max_iter=1000).get_params()


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_grid_search_over_the_estimator_including_the_default(cls):
    x, y = table()
    grid = {"estimator": [None, LogisticRegression(C=1.0)]}
    search = GridSearchCV(cls(k=15), grid, cv=2).fit(x, y)
    assert len(search.cv_results_["params"]) == 2
    assert not np.isnan(search.cv_results_["mean_test_score"]).any()


# --- the reduction is fixed by the class --------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "other"),
    [(EPR, WEPRPooling()), (WEPR, EPRPooling()), (EPRClassifier, WEPRPooling()), (WEPRClassifier, EPRPooling())],
    ids=ESTIMATOR_IDS,
)
def test_fit_rejects_an_entropy_pooling_the_class_does_not_fix(cls, other):
    x, y = fit_data(cls)
    detector = cls(k=15).set_params(entropy__pooling=other)
    with pytest.raises(ValueError, match=cls.__name__) as err:
        detector.fit(x, y)
    assert f"pooling={other!r}" in str(err.value)


@pytest.mark.parametrize("cls", [EPRClassifier, WEPRClassifier])
@pytest.mark.parametrize("step", ["passthrough", StandardScaler()], ids=["passthrough", "no-k"])
def test_fit_rejects_a_first_step_without_k(cls, step):
    x, y = table()
    classifier = cls(k=15).set_params(entropy=step)
    with pytest.raises(ValueError, match=r"'entropy' step .* has no `k`"):
        classifier.fit(x, y)


@pytest.mark.parametrize(("cls", "other"), [(EPRClassifier, WEPRPooling()), (WEPRClassifier, EPRPooling())])
def test_the_way_out_the_pooling_error_names_takes_arrays(cls, other):
    x, y = table()
    with pytest.raises(ValueError, match="a subclass that sets `pooling`"):
        cls(k=15).set_params(entropy__pooling=other).fit(x, y)

    class Other(cls):
        pooling = other

    assert Other(k=15).fit(x, y).predict_proba(x).shape == (len(x), 2)


# --- feature names and set_output -------------------------------------------------------


K = 15


def fitted(cls):
    """`cls` fitted at `K` ranks: on a random table for a classifier, on the responses for a detector."""
    return cls(k=K).fit(*fit_data(cls, width=K))


def front_input(cls):
    """What the steps before the classifier take: the responses for a detector, a table otherwise."""
    return RESPONSES if cls in DETECTORS else table(width=K)[0]


def entropy_input(cls):
    """What the entropy step takes: a cube the parser built, or the table itself."""
    return fitted(cls)[:1].transform(RESPONSES) if cls in DETECTORS else table(width=K)[0]


def entropy_slice(estimator):
    """The entropy step alone, as a one-step `Pipeline`."""
    return estimator[-2:-1]


@pytest.mark.parametrize("cls", [EPR, EPRClassifier], ids=["EPR", "EPRClassifier"])
def test_epr_names_its_one_feature(cls):
    assert list(cls(k=K)[:-1].get_feature_names_out()) == ["epr"]


@pytest.mark.parametrize("cls", [WEPR, WEPRClassifier], ids=["WEPR", "WEPRClassifier"])
def test_wepr_names_mean_then_max_per_rank(cls):
    names = list(cls(k=K)[:-1].get_feature_names_out())
    assert names == [f"wepr_{b}_rank{i}" for b in ("mean", "max") for i in range(K)]


def test_the_entropy_step_names_its_features_from_k_alone():
    assert len(entropy_slice(WEPR(k=K)).get_feature_names_out()) == 2 * K


@pytest.mark.parametrize("pooling", [EPRPooling(), WEPRPooling()], ids=["EPR", "WEPR"])
def test_the_names_of_a_narrow_table_match_the_columns_it_produces(pooling):
    """Names follow the fitted width, not `k`, for a table narrower than `k`."""
    step = EntropyTransformer(pooling=pooling, k=K).fit(table(n=6, width=3)[0])
    features = step.transform(table(n=6, width=3)[0])
    assert len(step.get_feature_names_out()) == features.shape[1]
    assert len(step.set_output(transform="pandas").transform(table(n=6, width=3)[0]).columns) == features.shape[1]


def test_the_names_of_a_narrow_wepr_table_are_those_of_its_own_ranks():
    step = EntropyTransformer(pooling=WEPRPooling(), k=K).fit(table(n=6, width=3)[0])
    expected = [f"wepr_{b}_rank{i}" for b in ("mean", "max") for i in range(3)]
    assert list(step.get_feature_names_out()) == expected


def test_a_refit_on_a_cube_names_the_cube_width_not_the_stale_table_width():
    step = EntropyTransformer(pooling=WEPRPooling(), k=K).fit(table(n=6, width=3)[0])
    assert len(step.get_feature_names_out()) == 6
    step.fit(-np.ones((2, 4, K)))
    assert len(step.get_feature_names_out()) == 2 * K


def test_a_refit_on_a_cube_forgets_the_column_names_of_the_table():
    frame = pd.DataFrame(table(n=6, width=3)[0], columns=list("abc"))
    step = EntropyTransformer(pooling=WEPRPooling(), k=3).fit(frame).fit(-np.ones((2, 4, 3)))
    assert not hasattr(step, "feature_names_in_")
    assert list(step.get_feature_names_out()) == [f"wepr_{b}_rank{i}" for b in ("mean", "max") for i in range(3)]


def test_input_features_of_a_table_wider_than_k_name_its_first_k_columns():
    step = EntropyTransformer(pooling=WEPRPooling(), k=2).fit(table(n=6, width=4)[0])
    assert step.get_feature_names_out(list("abcd")).tolist() == [f"wepr_{b}_{r}" for b in ("mean", "max") for r in "ab"]
    with pytest.raises(ValueError, match="input_features"):
        step.get_feature_names_out(list("ab"))


def test_input_features_are_checked_against_the_fitted_width_when_narrower_than_k():
    step = EntropyTransformer(pooling=WEPRPooling(), k=K).fit(table(n=6, width=3)[0])
    assert step.get_feature_names_out(["a", "b", "c"]).tolist() == [
        f"wepr_{b}_{r}" for b in ("mean", "max") for r in "abc"
    ]
    with pytest.raises(ValueError, match="input_features"):
        step.get_feature_names_out([f"r{i}" for i in range(K)])


def test_a_narrow_named_frame_names_the_columns_it_produces():
    frame = pd.DataFrame(table(n=6, width=3)[0], columns=list("abc"))
    step = EntropyTransformer(pooling=WEPRPooling(), k=K).fit(frame).set_output(transform="pandas")
    assert list(step.transform(frame).columns) == list(step.get_feature_names_out())


@pytest.mark.parametrize(("fitted", "given"), [(3, K), (K, 3)], ids=["wider", "narrower"])
def test_a_cube_of_another_width_than_the_fit_is_refused_at_transform(fitted, given):
    step = EntropyTransformer(pooling=WEPRPooling(), k=K).fit(-np.ones((2, 4, fitted)))
    with pytest.raises(ValueError, match=rf"X has {given} features, but EntropyTransformer is expecting {fitted}"):
        step.transform(-np.ones((2, 4, given)))


def test_an_empty_cube_is_not_refused_by_a_fitted_width():
    step = EntropyTransformer(pooling=WEPRPooling(), k=K).fit(-np.ones((2, 4, K)))
    assert step.validate(np.empty((0, 0, 0))).shape == (0, 0, 0)


def test_a_cube_fit_with_no_k_names_every_rank_it_was_fitted_on():
    step = EntropyTransformer(pooling=WEPRPooling(), k=None).fit(-np.ones((2, 4, 3)))
    assert len(step.get_feature_names_out()) == 6


def test_the_names_agree_with_the_coefficients_a_classifier_of_that_k_holds():
    """A pretrained file holds an unnamed `LogisticRegression`, so names are checked by count and order.

    Columns reach the classifier in the order of the names, so a coefficient position
    means the rank-wise feature named at that position.
    """
    x, y = table(width=5)
    classifier = WEPRClassifier(k=5).fit(x, y)
    names = classifier[:-1].get_feature_names_out()
    assert len(names) == classifier[-1].coef_.shape[1]
    features = classifier[:-1].set_output(transform="pandas").transform(x)
    assert list(features.columns) == list(names)


def test_input_features_must_match_k():
    step = WEPRClassifier(k=K).named_steps["entropy"]
    assert len(step.get_feature_names_out([f"r{i}" for i in range(K)])) == 2 * K
    with pytest.raises(ValueError, match="input_features"):
        step.get_feature_names_out(["a", "b"])


def test_wepr_with_no_k_cannot_name_features_unless_shown_the_rank_names():
    step = WEPRClassifier(k=K).named_steps["entropy"].set_params(k=None)
    with pytest.raises(ValueError, match="k"):
        step.get_feature_names_out()
    assert step.get_feature_names_out(["a", "b"]).tolist() == ["wepr_mean_a", "wepr_mean_b", "wepr_max_a", "wepr_max_b"]


@pytest.mark.parametrize("cls", [WEPR, WEPRClassifier], ids=["WEPR", "WEPRClassifier"])
def test_wepr_names_follow_the_column_order_of_the_features(cls):
    """The mean branch comes first: per rank, a mean never exceeds the max it is named beside."""
    pd = pytest.importorskip("pandas")
    estimator = fitted(cls)
    features = estimator[:-1].set_output(transform="pandas").transform(front_input(cls))
    assert is_bearable(features, pd.DataFrame)
    means = features.filter(like="wepr_mean_").to_numpy()
    maxes = features.filter(like="wepr_max_").to_numpy()
    assert list(features.columns) == list(estimator[:-1].get_feature_names_out())
    np.testing.assert_array_equal(features.to_numpy(), np.hstack([means, maxes]))
    assert np.all(means <= maxes + 1e-12)
    assert features.shape[1] == 2 * K


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_the_front_of_an_estimator_returns_a_named_frame(cls, backend):
    pytest.importorskip(backend)
    names = list(cls(k=K)[:-1].get_feature_names_out())
    for estimator in (cls(k=K), fitted(cls)):
        out = estimator[:-1].set_output(transform=backend).transform(front_input(cls))
        assert type(out).__module__.startswith(backend)
        assert list(out.columns) == names


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_the_entropy_slice_alone_returns_a_frame(cls):
    pytest.importorskip("pandas")
    estimator = fitted(cls)
    out = entropy_slice(estimator).set_output(transform="pandas").transform(entropy_input(cls))
    assert list(out.columns) == list(estimator[:-1].get_feature_names_out())


@pytest.mark.parametrize("cls", [WEPR, WEPRClassifier], ids=["WEPR", "WEPRClassifier"])
def test_transform_tokens_stays_three_dimensional_under_set_output(cls):
    pytest.importorskip("pandas")
    entropy = fitted(cls).named_steps["entropy"].set_output(transform="pandas")
    assert entropy.transform_tokens(entropy_input(cls)).ndim == 3


def test_the_parser_is_not_wrapped_into_a_frame():
    pytest.importorskip("pandas")
    cube = fitted(WEPR)[:1].set_output(transform="pandas").transform(RESPONSES)
    assert cube.ndim == 3


@pytest.mark.parametrize("cls", [EPR, EPRClassifier], ids=["EPR", "EPRClassifier"])
def test_a_custom_pooling_names_its_features_after_itself(cls):
    def mean_of_ranks(s, axis):
        return np.nanmean(np.nanmean(s, axis=-1, keepdims=True), axis=axis)

    estimator = fitted(cls)
    estimator.named_steps["entropy"].pooling = mean_of_ranks
    assert list(estimator[:-1].get_feature_names_out()) == ["mean_of_ranks"]


@pytest.mark.parametrize("cls", [EPR, EPRClassifier], ids=["EPR", "EPRClassifier"])
def test_set_output_rejects_unknown_formats_when_the_output_is_built(cls):
    front = fitted(cls)[:-1].set_output(transform="numpy")
    with pytest.raises(ValueError, match="numpy"):
        front.transform(front_input(cls))


def test_a_classifier_fitted_on_a_frame_names_its_features_after_the_columns():
    columns = ["a", "b", "c"]
    x, y = table(width=3)
    classifier = WEPRClassifier(k=3).fit(pd.DataFrame(x, columns=columns), y)
    assert list(classifier[:-1].get_feature_names_out()) == [
        f"wepr_{branch}_{name}" for branch in ("mean", "max") for name in columns
    ]
    with pytest.raises(ValueError, match="feature_names_in_"):
        classifier[:-1].get_feature_names_out(["x", "y", "z"])


def test_a_classifier_fitted_on_an_array_names_its_features_after_the_ranks():
    x, y = table(width=3)
    classifier = WEPRClassifier(k=3).fit(x, y)
    assert list(classifier[:-1].get_feature_names_out()) == [
        f"wepr_{branch}_rank{i}" for branch in ("mean", "max") for i in range(3)
    ]


# --- k reaches every step ----------------------------------------------------------------


@pytest.mark.parametrize("cls", [WEPR, WEPRClassifier], ids=["WEPR", "WEPRClassifier"])
def test_set_params_k_reaches_every_step_that_takes_it(cls):
    estimator = cls(k=K).set_params(k=5)
    assert estimator.k == estimator.named_steps["entropy"].k == 5
    if cls in DETECTORS:
        assert estimator.named_steps["parser"].k == 5
    assert len(estimator[:-1].get_feature_names_out()) == 10


@pytest.mark.parametrize("cls", [WEPR, WEPRClassifier], ids=["WEPR", "WEPRClassifier"])
def test_clone_keeps_k_on_every_step(cls):
    copy = clone(fitted(cls).set_params(k=7))
    assert copy.k == copy.named_steps["entropy"].k == 7
    if cls in DETECTORS:
        assert copy.named_steps["parser"].k == 7


def test_a_step_level_k_narrower_than_the_cube_keeps_the_first_ranks():
    detector = WEPR(k=K).set_params(entropy__k=5)
    assert detector[:-1].transform(RESPONSES).shape[1] == 10


def test_a_parser_level_k_that_disagrees_cannot_be_named_into_a_frame():
    pytest.importorskip("pandas")
    detector = WEPR(k=K).set_params(parser__k=5)
    with pytest.raises(ValueError, match="10"):
        detector[:-1].set_output(transform="pandas").transform(RESPONSES)


# --- a classifier takes arrays and a detector takes responses ----------------------------


@pytest.mark.parametrize("cls", DETECTORS, ids=IDS)
def test_a_detector_refuses_a_table_and_names_the_classifiers(cls):
    x, y = table()
    with pytest.raises(TypeError, match="EPRClassifier or WEPRClassifier"):
        cls(k=15).fit(x, y)
    with pytest.raises(TypeError, match="EPRClassifier or WEPRClassifier"):
        fitted(cls).predict_proba(x)


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_a_classifier_scores_a_padded_cube_and_locates_the_tokens(cls):
    cube = -np.random.RandomState(0).exponential(1.0, size=(6, 5, K))
    cube[:3, 3:] = np.nan
    classifier = cls(k=15).fit(cube, np.arange(6) % 2)
    assert classifier.predict_proba(cube).shape == (6, 2)
    per_token = classifier.predict_token_proba(cube)
    assert per_token.shape == (6, 5, 1)
    assert np.isnan(per_token[:3, 3:]).all()
    assert not np.isnan(per_token[:3, :3]).any()


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_a_grid_search_over_k_and_the_estimator_runs_on_a_cube_wider_than_k(cls):
    cube = -np.random.RandomState(1).exponential(1.0, size=(30, 4, K))
    y = (cube[:, 0, 0] > np.median(cube[:, 0, 0])).astype(int)
    grid = {"k": [3, 8], "estimator__C": [0.1, 1.0]}
    search = GridSearchCV(cls(estimator=LogisticRegression(), k=15), grid, cv=3).fit(cube, y)
    assert set(search.best_params_) == set(grid)
    assert search.best_estimator_.predict_proba(cube).shape == (30, 2)


def test_an_explicitly_default_estimator_is_reported_as_none_and_clones():
    detector = EPR(estimator=LogisticRegression(C=np.inf, max_iter=1000), k=15)
    assert detector.get_params()["estimator"] is None
    assert clone(detector).get_params()["estimator"] is None


def test_the_detectors_do_not_inherit_from_the_classifiers():
    for detector, classifier in zip(DETECTORS, CLASSIFIERS, strict=True):
        assert not issubclass(detector, classifier)


# --- package root exports ----------------------------------------------------------------


def test_the_estimators_are_exported_from_the_package_root_and_from_scoring():
    import artefactual
    import artefactual.scoring

    for cls in ESTIMATORS:
        assert vars(artefactual)[cls.__name__] is cls
        assert vars(artefactual.scoring)[cls.__name__] is cls
    assert {cls.__name__ for cls in ESTIMATORS} <= set(artefactual.__all__)


# --- slicing -----------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
@pytest.mark.parametrize("ind", [slice(None, None, 2), slice(None, None, -1)])
def test_a_slice_with_a_step_is_refused_as_pipeline_refuses_it(cls, ind):
    with pytest.raises(ValueError, match="step of 1"):
        cls(k=15)[ind]


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_a_slice_is_a_plain_pipeline_over_the_same_steps(cls):
    detector = cls(k=15)
    front = detector[:-1]
    assert type(front) is Pipeline
    assert front.steps == detector.steps[:-1]
    assert detector["classifier"] is detector.estimator
    assert detector[0] is detector.steps[0][1]


# --- n_features_in_ ----------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_n_features_in_is_the_width_of_the_fitted_table_and_absent_before(cls):
    detector = cls(k=15)
    with pytest.raises(AttributeError):
        detector.n_features_in_  # ruff: ignore[useless-expression]
    x, y = table(width=6)
    assert detector.fit(x, y).n_features_in_ == 6


# --- parameter validation ----------------------------------------------------------------


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
@pytest.mark.parametrize("k", [0, -2, 1.5, "3", None])
def test_fit_rejects_an_invalid_k_before_doing_anything_else(cls, k):
    x, y = fit_data(cls)
    detector = cls(k=15)
    detector.k = k
    with pytest.raises(ValueError, match="'k'"):
        detector.fit(x, y)


@pytest.mark.parametrize("cls", ESTIMATORS, ids=ESTIMATOR_IDS)
def test_fit_rejects_an_estimator_without_predict_proba(cls):
    x, y = fit_data(cls)
    from sklearn.svm import LinearSVC

    with pytest.raises(ValueError, match="'estimator'"):
        cls(estimator=LinearSVC(), k=15).fit(x, y)


# --- labels ------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", CLASSIFIERS, ids=CLASSIFIER_IDS)
def test_any_two_labels_are_accepted_and_the_second_is_column_one(cls):
    x, y = table()
    detector = cls(k=15).fit(x, y + 1)
    assert list(detector.classes_) == [1, 2]
    zero_one = cls(k=15).fit(x, y)
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
        "EPR", "WEPR", "EPRClassifier", "WEPRClassifier", "Judgment", "LogProbParser", "EntropyTransformer",
        "ChatCompletion", "ResponsesPayload", "BatchRequestOutput", "read_batch", "epr_reduction", "wepr_reduction",
    }  # fmt: skip
    assert names <= set(artefactual.__all__)
    assert "LogprobsMixin" not in artefactual.__all__


# --- a pooling swapped after the fit -------------------------------------------------------


@pytest.mark.parametrize("method", ["predict_proba", "predict_token_proba"])
def test_a_pooling_swapped_after_the_fit_is_refused_with_a_request_to_refit(method):
    x, y = table(width=5)
    classifier = EPRClassifier(k=5).fit(x, y)
    classifier.set_params(entropy__pooling=WEPRPooling())
    with pytest.raises(ValueError, match="refit"):
        {"predict_proba": classifier.predict_proba, "predict_token_proba": classifier.predict_token_proba}[method](x)


def test_set_params_alone_does_not_refit_or_raise():
    x, y = table(width=5)
    classifier = EPRClassifier(k=5).fit(x, y)
    classifier.set_params(entropy__pooling=WEPRPooling())
    assert classifier.get_params()["entropy__pooling"] == WEPRPooling()


def test_a_step_never_fitted_transforms_with_any_pooling():
    x, _ = table(width=5)
    assert EntropyTransformer(pooling=WEPRPooling(), k=5).transform(x).shape == (len(x), 10)


def test_a_pooling_mutated_after_the_fit_is_refused_too():
    x, _ = table(width=5)
    pooling = WEPRPooling()
    step = EntropyTransformer(pooling=pooling, k=5).fit(x)
    pooling.extra = 2  # any state a pooling carries is part of what was fitted
    with pytest.raises(ValueError, match="refit"):
        step.transform(x)


class ScaledMean(Pooling):
    """One mean per rank, scaled by an array."""

    constant_width = False  # `scale` fits one `k`, so the count cannot be probed at others

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
        return np.nanmean(X * self.scale, axis=axis)

    def mean(self, x, axis):
        return np.nanmean(x, axis=axis)


def rank_percentile(x, axis, q):
    return np.nanpercentile(x, q, axis=axis)


class MeanCallable:
    """A callable object that defines no equality."""

    def __call__(self, x, axis):
        return np.nanmean(x, axis=axis)


class ScaledBy(Pooling):
    """Holds a callable as an attribute and applies it."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
        return self.fn(X, axis=axis)


CUSTOM_POOLINGS = {
    "partial": partial(rank_percentile, q=95),
    "callable-object": MeanCallable(),
    "bound-method": ScaledMean(1.0).mean,
    "array-attribute": ScaledMean(np.ones(5)),
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
    step = EntropyTransformer(pooling=pooling, k=5).fit(x)
    np.testing.assert_array_equal(copied(step).transform(x), step.transform(x))


@pytest.mark.parametrize("pooling", CUSTOM_POOLINGS.values(), ids=CUSTOM_POOLINGS.keys())
def test_a_detector_over_any_pooling_fits_and_scores(pooling):
    responses = RESPONSES * 4
    y = np.arange(len(responses)) % 2
    detector = PoolingDetector(pooling, k=5).fit(responses, y)
    assert clone(detector).fit(responses, y).predict_proba(responses).shape == (len(responses), 2)
