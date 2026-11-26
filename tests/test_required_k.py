"""An EPR detector loaded from weights needs the `k` they were trained at, so none is guessed."""

import pytest
from conftest import fitted_logistic

from artefactual import EPR, WEPR


@pytest.mark.parametrize("identifier", ["artefactory/does-not-exist", "/no/such/model.skops"])
def test_epr_from_pretrained_without_k_is_refused_before_any_lookup(identifier):
    with pytest.raises(TypeError, match="pass the k the model was trained with"):
        EPR.from_pretrained(identifier)


@pytest.mark.parametrize("k", [3, 15, 40])
def test_epr_accepts_weights_at_any_k(k):
    # The file holds no `k`: EPR pools to one feature at every `k`, so a wrong `k` cannot be told from the right one.
    assert EPR.from_estimator(fitted_logistic(0.0, [1.0]), "somewhere", k=k).k == k


def test_wepr_takes_no_k():
    with pytest.raises(TypeError, match="'k'"):
        WEPR.from_estimator(fitted_logistic(0.0, [1.0] * 30), "somewhere", k=15)
