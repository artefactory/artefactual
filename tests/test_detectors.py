"""`EPR` and `WEPR` as plain scikit-learn classifiers, loaded, cloned and composed."""

from operator import methodcaller

import numpy as np
import pytest
from conftest import fitted_logistic
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import InvalidParameterError
from test_common import MyDetector

from artefactual import EPR, WEPR, ChatCompletion, DetectorPipeline, LogProbParser, make_detector
from artefactual.scoring import EPRPooling, TokenProbaMixin, token_proba
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer

WIDTH = 15
RNG = np.random.RandomState(0)
CUBE = -RNG.exponential(1.0, size=(40, 5, WIDTH))
LABELS = RNG.randint(0, 2, size=40)


def loaded_epr(tmp_path):
    """An EPR carrying published weights, written and read back as a user would."""
    trained = EPR(k=WIDTH).fit(CUBE, LABELS)
    return EPR.from_pretrained(trained.save_pretrained(tmp_path / "epr.skops"), k=WIDTH)


def test_the_loaded_detector_is_fitted_with_a_frozen_estimator(tmp_path):
    loaded = loaded_epr(tmp_path)

    assert type(loaded.estimator) is FrozenEstimator
    assert loaded.predict_proba(CUBE).shape == (40, 2)


def test_a_clone_of_the_loaded_detector_predicts_the_same(tmp_path):
    loaded = loaded_epr(tmp_path)

    np.testing.assert_array_equal(clone(loaded).predict_proba(CUBE), loaded.predict_proba(CUBE))


def test_a_clone_of_the_loaded_epr_keeps_k(tmp_path):
    assert clone(loaded_epr(tmp_path)).k == WIDTH


def test_a_pipeline_around_a_clone_of_the_loaded_detector_predicts(tmp_path):
    loaded = loaded_epr(tmp_path)
    detector = make_pipeline(EntropyTransformer(EPRPooling()), clone(loaded.estimator))

    assert detector.predict_proba(CUBE).shape == (40, 2)


def test_cross_validation_of_a_loaded_pipeline_leaves_the_weights_alone(tmp_path):
    loaded = loaded_epr(tmp_path)
    before = loaded.estimator.coef_.copy()

    cross_val_score(loaded, CUBE, LABELS, cv=3)

    np.testing.assert_array_equal(loaded.estimator.coef_, before)


def test_fit_on_a_loaded_detector_keeps_the_weights(tmp_path):
    loaded = loaded_epr(tmp_path)
    before = loaded.predict_proba(CUBE)

    loaded.fit(CUBE, 1 - LABELS)

    np.testing.assert_array_equal(loaded.predict_proba(CUBE), before)


def fitted_epr(_tmp_path):
    return EPR(k=WIDTH).fit(CUBE, LABELS)


@pytest.mark.parametrize("make", [fitted_epr, loaded_epr, lambda tmp_path: clone(loaded_epr(tmp_path))])
@pytest.mark.parametrize("method", ["predict", "predict_proba", "decision_function", "predict_token_proba"])
def test_epr_with_k_refuses_another_width_and_names_the_remedy(make, method, tmp_path):
    detector = make(tmp_path)

    with pytest.raises(ValueError, match=r"trained on 15 candidates per token, got 10: use LogProbParser\(k=15\)"):
        methodcaller(method, CUBE[..., :10])(detector)


def test_fit_refuses_another_width_and_names_the_remedy():
    with pytest.raises(ValueError, match=r"got 10: use LogProbParser\(k=15\)"):
        EPR(k=15).fit(CUBE[..., :10], LABELS)


@pytest.mark.parametrize("k", [0, "a"])
def test_epr_refuses_an_invalid_k_at_fit(k):
    with pytest.raises(InvalidParameterError):
        EPR(k=k).fit(CUBE, LABELS)


def test_wepr_refuses_an_estimator_without_predict_proba():
    with pytest.raises(InvalidParameterError):
        WEPR(estimator=LinearRegression()).fit(CUBE, LABELS)


def fitted_wepr(_tmp_path):
    return WEPR().fit(CUBE, LABELS)


def fitted_epr_without_k(_tmp_path):
    return EPR().fit(CUBE, LABELS)


RESPONSES = [[{"choices": []}], {"choices": []}, ChatCompletion.model_construct()]
FITTED_DETECTORS = [fitted_epr, fitted_wepr, fitted_epr_without_k, loaded_epr]


@pytest.mark.parametrize("responses", RESPONSES)
@pytest.mark.parametrize("make", FITTED_DETECTORS)
@pytest.mark.parametrize("method", ["predict", "predict_proba", "decision_function", "predict_token_proba"])
def test_responses_are_refused_with_the_pipeline_fix(make, method, responses, tmp_path):
    with pytest.raises(TypeError, match="is_logprob_array"):
        methodcaller(method, responses)(make(tmp_path))


@pytest.mark.parametrize("responses", RESPONSES)
@pytest.mark.parametrize("detector", [EPR(), WEPR(), EPR(k=WIDTH)])
def test_fit_refuses_responses_before_anything_else(detector, responses):
    with pytest.raises(TypeError, match="is_logprob_array"):
        detector.fit(responses, LABELS)


@pytest.mark.parametrize("detector", [EPR(), WEPR(), EPR(k=WIDTH)])
def test_a_fitted_detector_keeps_the_width_and_feature_names_of_its_entropy_step(detector):
    fitted = detector.fit(CUBE, LABELS)

    assert fitted.n_features_in_ == WIDTH
    assert len(fitted.get_feature_names_out()) == len(fitted.pipeline_[-2].get_feature_names_out())


def test_wepr_has_no_k():
    with pytest.raises(TypeError):
        WEPR(k=15)  # ty: ignore[unknown-argument]


def test_the_passed_estimator_is_never_fitted_in_place():
    estimator = fitted_logistic(0.0, [1.0])
    classifier = WEPR(estimator=clone(estimator)).fit(CUBE, LABELS)

    assert classifier.estimator.get_params() == estimator.get_params()
    assert "coef_" in vars(classifier.pipeline_[-1])


def test_a_custom_detector_scores_tokens():
    detector = MyDetector().fit(CUBE, LABELS)
    scores = detector.predict_token_proba(CUBE)

    assert scores.shape == (40, 5, 1)
    assert not np.isnan(scores).any()


@pytest.mark.parametrize("detector", [EPR(), WEPR()])
def test_token_scores_leave_padding_as_nan(detector):
    padded = CUBE.copy()
    padded[:, 3:] = np.nan

    scores = detector.fit(CUBE, LABELS).predict_token_proba(padded)

    assert np.isnan(scores[:, 3:]).all()
    assert not np.isnan(scores[:, :3]).any()


def test_a_loaded_detector_knows_its_classes_before_any_predict(tmp_path):
    loaded = loaded_epr(tmp_path)

    np.testing.assert_array_equal(loaded.classes_, loaded.estimator.estimator.classes_)
    np.testing.assert_array_equal(clone(loaded).classes_, loaded.classes_)


def test_an_unfitted_detector_has_no_classes():
    with pytest.raises(AttributeError):
        EPR().classes_  # ruff: ignore[useless-expression]
    with pytest.raises(AttributeError):
        WEPR().pipeline_  # ruff: ignore[useless-expression]


def test_from_estimator_freezes_a_fitted_classifier():
    detector = WEPR.from_estimator(fitted_logistic(0.0, [1.0] * 2 * WIDTH), "somewhere")

    assert type(detector.estimator) is FrozenEstimator
    assert detector.predict_proba(CUBE).shape == (40, 2)


def test_from_estimator_refuses_an_unfitted_classifier():
    with pytest.raises(ValueError, match="somewhere"):
        WEPR.from_estimator(LogisticRegression(), "somewhere")


@pytest.mark.parametrize("make", [lambda loaded: loaded, clone])
def test_a_loaded_detector_scores_tokens_without_fit(tmp_path, make):
    scores = make(loaded_epr(tmp_path)).predict_token_proba(CUBE)

    assert scores.shape == (40, 5, 1)
    assert not np.isnan(scores).any()


def test_make_detector_requires_k():
    with pytest.raises(TypeError):
        make_detector()  # ty: ignore[missing-argument]


def test_make_detector_fit_refuses_an_invalid_k():
    with pytest.raises(InvalidParameterError, match="'k'"):
        make_detector(k=0).fit([], [])


def test_make_detector_refuses_a_positional_call():
    with pytest.raises(TypeError):
        make_detector(15)  # ty: ignore[missing-argument, too-many-positional-arguments]


def test_make_detector_refuses_a_positional_classifier():
    with pytest.raises(TypeError):
        make_detector(WEPR(), k=15)  # ty: ignore[missing-argument, too-many-positional-arguments]


def test_make_detector_takes_the_classifier_by_keyword_and_names_the_steps():
    classifier = EPR(k=15)
    detector = make_detector(k=15, classifier=classifier)

    assert detector[-1] is classifier
    assert [name for name, _ in detector.steps] == ["logprobparser", "epr"]


def test_make_detector_defaults_to_wepr_behind_a_parser():
    detector = make_detector(k=15)

    assert type(detector) is DetectorPipeline
    assert type(detector[0]) is LogProbParser
    assert detector[0].k == 15
    assert type(detector[-1]) is WEPR


@pytest.mark.parametrize("make", [lambda loaded: loaded, clone])
def test_a_loaded_detector_reports_no_n_features_in(tmp_path, make):
    loaded = make(loaded_epr(tmp_path))

    with pytest.raises(AttributeError):
        loaded.n_features_in_  # ruff: ignore[useless-expression]


@pytest.mark.parametrize("detector", [EPR(), WEPR()])
def test_a_fitted_detector_reports_the_width_of_the_cube_it_fitted_on(detector):
    assert detector.fit(CUBE, LABELS).n_features_in_ == WIDTH


def chat(ranks):
    """A one-choice completion response whose every token has `ranks` as its candidates."""
    token = {"token": "t", "logprob": ranks[0], "top_logprobs": [{"logprob": r} for r in ranks]}
    return {"choices": [{"logprobs": {"content": [token] * 3}}]}


def test_a_grid_search_tunes_the_parser_and_the_classifier_of_a_new_detector():
    responses = [chat(list(-RNG.exponential(1.0 + label, size=10))) for label in LABELS[:20]]
    search = GridSearchCV(
        make_pipeline(LogProbParser(), WEPR(estimator=LogisticRegression())),
        {"logprobparser__k": [5, 10], "wepr__estimator__C": [0.1, 1]},
        cv=2,
    ).fit(responses, LABELS[:20])

    assert search.best_params_["logprobparser__k"] in (5, 10)
    assert search.best_estimator_.predict_proba(responses).shape == (20, 2)


def chats(n, width):
    """`n` completion responses whose tokens each have `width` candidates, of varying lengths."""
    return [
        {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "t",
                                "logprob": -1.0,
                                "top_logprobs": [{"logprob": float(r)} for r in -RNG.exponential(1.0, size=width)],
                            }
                        ]
                        * (2 + i % 3)
                    }
                }
            ]
        }
        for i in range(n)
    ]


def test_the_pipeline_scores_tokens_as_its_classifier_scores_the_parsed_responses(tmp_path):
    responses = chats(12, WIDTH)
    parser = LogProbParser(k=WIDTH)
    fitted_wepr = WEPR().fit(parser.transform(responses), LABELS[:12])

    for classifier in (fitted_wepr, loaded_epr(tmp_path)):
        pipeline = make_detector(k=WIDTH, classifier=classifier)

        np.testing.assert_allclose(
            pipeline.predict_token_proba(responses),
            classifier.predict_token_proba(parser.transform(responses)),
            equal_nan=True,
        )


def test_a_fitted_default_pipeline_scores_tokens_with_nan_at_padding():
    responses = chats(12, WIDTH)
    scores = make_detector(k=WIDTH).fit(responses, LABELS[:12]).predict_token_proba(responses)

    assert scores.shape == (12, 4, 1)
    assert np.isnan(scores).any()
    assert np.isnan(scores[0, 2:]).all()


def test_an_unfitted_pipeline_refuses_token_scores():
    with pytest.raises(NotFittedError):
        make_detector(k=WIDTH).predict_token_proba(chats(2, WIDTH))


def test_a_cloned_detector_pipeline_is_a_detector_pipeline_with_the_same_params():
    detector = make_detector(k=10, classifier=EPR(k=10))
    cloned = clone(detector)

    assert type(cloned) is DetectorPipeline
    assert cloned.get_params()["logprobparser__k"] == 10
    assert cloned.get_params()["epr__k"] == 10


def test_a_grid_search_tunes_the_parser_of_a_detector_pipeline():
    responses = [chat(list(-RNG.exponential(1.0 + label, size=10))) for label in LABELS[:20]]
    search = GridSearchCV(make_detector(k=5), {"logprobparser__k": [5, 10]}, cv=2).fit(responses, LABELS[:20])

    assert type(search.best_estimator_) is DetectorPipeline
    assert search.best_estimator_.predict_token_proba(responses).shape[0] == 20


def test_the_pipeline_still_refuses_a_wrong_width_classifier_for_token_scores(tmp_path):
    detector = make_detector(k=10, classifier=loaded_epr(tmp_path))

    with pytest.raises(ValueError, match=r"trained on 15 candidates per token, got 10: use LogProbParser\(k=15\)"):
        detector.predict_token_proba(chats(2, 10))


def padded_cube():
    cube = CUBE.copy()
    cube[:, 3:] = np.nan
    return cube


def flat_oracle(features, scorer):
    """Per-token scores laid out as `(n, tokens, 1)`, NaN where the features are padding."""
    n, tokens, width = features.shape
    flat = features.reshape(n * tokens, width)
    keep = ~np.isnan(flat).any(axis=1)
    out = np.full(n * tokens, np.nan)
    if keep.any():
        out[keep] = scorer(flat[keep])
    return out.reshape(n, tokens, 1)


@pytest.mark.parametrize("make", [fitted_epr, fitted_wepr, loaded_epr])
@pytest.mark.parametrize("batch", [CUBE, padded_cube(), np.empty((0, 0, WIDTH)), np.full((3, 4, WIDTH), np.nan)])
def test_token_scores_equal_the_classifier_on_the_pooled_tokens(make, batch, tmp_path):
    detector = make(tmp_path)
    entropy, classifier = detector.pipeline_[-2], detector.pipeline_[-1]
    n, tokens, _ = batch.shape
    expected = np.full((n, tokens, 1), np.nan)
    if n and tokens:
        expected = flat_oracle(entropy.transform_tokens(batch), lambda rows: classifier.predict_proba(rows)[:, 1])

    scores = detector.predict_token_proba(batch)

    np.testing.assert_array_equal(scores, expected)
    assert scores.shape == (n, tokens, 1)


def test_make_detector_scores_an_empty_batch():
    detector = make_detector(k=WIDTH, classifier=WEPR().fit(CUBE, LABELS))

    assert detector.predict_token_proba([]).shape == (0, 0, 1)


def test_a_custom_pipeline_scores_tokens_through_its_steps():
    pipeline = DetectorPipeline([
        ("entropy", EntropyTransformer(EPRPooling())),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ]).fit(CUBE, LABELS)
    entropy, scaler, classifier = (step for _, step in pipeline.steps)
    expected = flat_oracle(
        entropy.transform_tokens(CUBE), lambda rows: classifier.predict_proba(scaler.transform(rows))[:, 1]
    )

    scores = pipeline.predict_token_proba(CUBE)

    np.testing.assert_array_equal(scores, expected)
    assert scores.shape == (40, 5, 1)


def test_make_detector_refuses_a_classifier_without_token_scores():
    with pytest.raises(TypeError):
        make_detector(k=WIDTH, classifier=LogisticRegression())  # ty: ignore[invalid-argument-type]


def test_a_detector_pipeline_without_a_token_step_or_token_classifier_refuses_to_fit():
    with pytest.raises(TypeError, match="TokenTransformer"):
        DetectorPipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())]).fit(CUBE[:, 0], LABELS)


def test_a_detector_pipeline_refuses_token_scores_from_a_three_class_last_step():
    pipeline = DetectorPipeline([
        ("entropy", EntropyTransformer(pooling=EPRPooling())),
        ("classifier", LogisticRegression()),
    ])
    pipeline.fit(CUBE, np.arange(len(CUBE)) % 3)
    with pytest.raises(TypeError, match="is_fitted_binary"):
        pipeline.predict_token_proba(CUBE)


class NoPipeline(TokenProbaMixin, BaseEstimator):
    """A fitted-looking class that never builds `pipeline_`."""

    def fit(self, y):
        self.classes_ = np.unique(y)
        return self


def test_the_mixin_on_a_class_without_a_pipeline_names_it():
    with pytest.raises(TypeError, match="pipeline_"):
        NoPipeline().fit(LABELS).predict_token_proba(CUBE)


def test_a_user_detector_scores_tokens_through_token_proba_of_its_pipeline():
    detector = MyDetector().fit(CUBE[:, 0], LABELS)
    np.testing.assert_array_equal(detector.predict_token_proba(CUBE), token_proba(detector.pipeline_.steps, CUBE))


def test_token_proba_refuses_a_three_class_last_step_called_directly():
    pipeline = make_pipeline(EntropyTransformer(pooling=EPRPooling()), LogisticRegression())
    pipeline.fit(CUBE, np.arange(len(CUBE)) % 3)
    with pytest.raises(TypeError, match="is_fitted_binary"):
        token_proba(pipeline.steps, CUBE)
