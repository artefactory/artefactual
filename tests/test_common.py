"""scikit-learn's own estimator checks, run against the detectors, the parser and the transformer.

No check is an expected failure: the detectors hold a pipeline in a fitted attribute and
leave their parameters alone.
"""

import dataclasses

import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import (
    check_estimator,
    check_estimators_overwrite_params,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
    parametrize_with_checks,
)
from sklearn.utils.validation import check_is_fitted

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring import EPR, WEPR, TokenProbaMixin
from artefactual.scoring.entropy_classifier import check_binary_target
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.entropy_methods.pooling import EPRPooling, WEPRPooling

# Never-vacuous guard: the checks that exercise what this package changed.
TRANSFORMER_MUST_RUN = {"check_estimators_pickle", "check_transformer_general", "check_fit2d_1feature"}
MUST_RUN = {"check_classifiers_train", "check_estimators_pickle", "check_fit2d_1feature"}


class MyDetector(TokenProbaMixin, ClassifierMixin, BaseEstimator):
    """A user-written detector: a custom pooling, its own fit and predict*, token scores from `TokenProbaMixin`."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = True
        tags.input_tags.three_d_array = False
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    def fit(self, X, y):  # ruff: ignore[invalid-argument-name]
        check_binary_target(y)
        self.pipeline_ = make_pipeline(
            EntropyTransformer(MyPooling()), clone(LogisticRegression() if self.estimator is None else self.estimator)
        ).fit(X, y)
        self.classes_ = self.pipeline_.classes_
        self.n_features_in_ = self.pipeline_[0].n_features_in_
        return self

    def predict(self, X):  # ruff: ignore[invalid-argument-name]
        check_is_fitted(self)
        return self.pipeline_.predict(X)

    def predict_proba(self, X):  # ruff: ignore[invalid-argument-name]
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)


@dataclasses.dataclass(frozen=True)
class MyPooling(EPRPooling):
    """EPR under another name: one feature per sequence."""


DETECTORS = [EPR(), WEPR(), MyDetector()]


@parametrize_with_checks(DETECTORS)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("classifier", DETECTORS, ids=lambda estimator: type(estimator).__name__)
def test_the_named_checks_ran_and_none_failed(classifier):
    """The suite is not vacuous: the checks that matter ran, and passed."""
    results = check_estimator(classifier, on_fail=None, expected_failed_checks=None)
    status = {r["check_name"]: r["status"] for r in results}
    assert all(status[name] == "passed" for name in MUST_RUN)
    assert "failed" not in status.values()
    assert sum(state == "passed" for state in status.values()) >= 50


TRANSFORMERS = [EntropyTransformer(pooling=EPRPooling()), EntropyTransformer(pooling=WEPRPooling())]

# The parser takes completion responses and refuses arrays, which every data-generating
# check sends, so only the checks that need no data are called by hand.
PARSER_CHECKS = [
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_get_params_invariance,
    check_set_params,
    check_estimators_overwrite_params,
]


@parametrize_with_checks(
    TRANSFORMERS,
    xfail_strict=True,
)
def test_sklearn_compatible_transformer(estimator, check):
    """Neither pooling is expected to fail a check, `check_estimators_nan_inf` included."""
    check(estimator)


@pytest.mark.parametrize("check", PARSER_CHECKS, ids=[check.__name__ for check in PARSER_CHECKS])
def test_sklearn_compatible_parser(check):
    check(type(LogProbParser(k=15)).__name__, LogProbParser(k=15))


@pytest.mark.parametrize("transformer", TRANSFORMERS, ids=["EPR", "WEPR"])
def test_the_transformer_checks_ran_and_none_failed(transformer):
    """The suite is not vacuous for the transformer: the named checks ran, and passed."""
    results = check_estimator(transformer, on_fail=None, expected_failed_checks=None)
    status = {r["check_name"]: r["status"] for r in results}
    assert all(status[name] == "passed" for name in TRANSFORMER_MUST_RUN | {"check_estimators_nan_inf"})
    assert "failed" not in status.values()
    assert sum(state == "passed" for state in status.values()) >= 25
