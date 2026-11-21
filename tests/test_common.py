"""scikit-learn's own estimator checks, run against the classifiers, the parser and the transformer.

The classifiers are `Pipeline` subclasses, so they inherit `Pipeline`'s one open
incompatibility: fitting overwrites the steps in place. scikit-learn lists the same
limitation for `Pipeline` itself in `sklearn/utils/_test_common/instance_generator.py`
(`PER_ESTIMATOR_XFAIL_CHECKS[Pipeline]`: "Pipeline changes the `steps` parameter, which it
shouldn't. Therefore this test is x-fail until we fix this."), for this check and for
`check_dont_overwrite_parameters`. The classifiers pass the latter, so only the first is
declared.
"""

import pytest
from sklearn.utils.estimator_checks import (
    check_estimator,
    check_estimators_overwrite_params,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
    parametrize_with_checks,
)

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring import EPRClassifier, WEPRClassifier
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.entropy_methods.pooling import EPRPooling, WEPRPooling

EXPECTED_FAILED_CHECKS = {
    "check_estimators_overwrite_params": (
        "Pipeline fits its steps in place, which this check reads as the constructor "
        "parameters being overwritten. scikit-learn lists the same check as an expected "
        "failure for Pipeline itself."
    ),
}

# Never-vacuous guard: the checks that exercise what this package changed.
TRANSFORMER_MUST_RUN = {"check_estimators_pickle", "check_transformer_general", "check_fit2d_1feature"}
MUST_RUN = {"check_classifiers_train", "check_estimators_pickle", "check_fit2d_1feature"}


@parametrize_with_checks(
    [EPRClassifier(k=15), WEPRClassifier(k=15)],
    expected_failed_checks=lambda estimator: EXPECTED_FAILED_CHECKS,  # ruff: ignore[unused-lambda-argument]
    # Strict: if the declared failure ever passes, the suite goes red and the entry must go.
    xfail_strict=True,
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize(
    "classifier", [EPRClassifier(k=15), WEPRClassifier(k=15)], ids=["EPRClassifier", "WEPRClassifier"]
)
def test_the_named_checks_ran_and_only_the_declared_one_failed(classifier):
    """The suite is not vacuous: the checks that matter ran, and passed."""
    results = check_estimator(classifier, on_fail=None, expected_failed_checks=None)
    status = {r["check_name"]: r["status"] for r in results}
    assert all(status[name] == "passed" for name in MUST_RUN)
    assert {name for name, state in status.items() if state == "failed"} == set(EXPECTED_FAILED_CHECKS)
    assert sum(state == "passed" for state in status.values()) >= 50


TRANSFORMERS = [EntropyTransformer(pooling=EPRPooling(), k=15), EntropyTransformer(pooling=WEPRPooling(), k=15)]

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
