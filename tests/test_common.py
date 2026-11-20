"""scikit-learn's own estimator checks, run against `EPRClassifier` and `WEPRClassifier`.

The classifiers are `Pipeline` subclasses, so they inherit `Pipeline`'s one open
incompatibility: fitting overwrites the steps in place. scikit-learn lists the same
limitation for `Pipeline` itself in `sklearn/utils/_test_common/instance_generator.py`
(`PER_ESTIMATOR_XFAIL_CHECKS[Pipeline]`: "Pipeline changes the `steps` parameter, which it
shouldn't. Therefore this test is x-fail until we fix this."), for this check and for
`check_dont_overwrite_parameters`. The classifiers pass the latter, so only the first is
declared.
"""

import pytest
from sklearn.utils.estimator_checks import check_estimator, parametrize_with_checks

from artefactual.scoring import EPRClassifier, WEPRClassifier

EXPECTED_FAILED_CHECKS = {
    "check_estimators_overwrite_params": (
        "Pipeline fits its steps in place, which this check reads as the constructor "
        "parameters being overwritten. scikit-learn lists the same check as an expected "
        "failure for Pipeline itself."
    ),
}

# Never-vacuous guard: the checks that exercise what this package changed.
MUST_RUN = {"check_classifiers_train", "check_estimators_pickle", "check_fit2d_1feature"}


@parametrize_with_checks(
    [EPRClassifier(), WEPRClassifier()],
    expected_failed_checks=lambda estimator: EXPECTED_FAILED_CHECKS,  # ruff: ignore[unused-lambda-argument]
    # Strict: if the declared failure ever passes, the suite goes red and the entry must go.
    xfail_strict=True,
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


@pytest.mark.parametrize("classifier", [EPRClassifier(), WEPRClassifier()], ids=["EPRClassifier", "WEPRClassifier"])
def test_the_named_checks_ran_and_only_the_declared_one_failed(classifier):
    """The suite is not vacuous: the checks that matter ran, and passed."""
    results = check_estimator(classifier, on_fail=None, expected_failed_checks=None)
    status = {r["check_name"]: r["status"] for r in results}
    assert all(status[name] == "passed" for name in MUST_RUN)
    assert {name for name, state in status.items() if state == "failed"} == set(EXPECTED_FAILED_CHECKS)
    assert sum(state == "passed" for state in status.values()) >= 50
