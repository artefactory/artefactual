"""`k` is the width the weights were trained at, so no estimator picks one for you."""

import numpy as np
import pytest
from conftest import fitted_logistic

from artefactual import (
    EPR,
    WEPR,
    EntropyTransformer,
    EPRClassifier,
    EPRPooling,
    PoolingDetector,
    WEPRClassifier,
    WEPRPooling,
)


@pytest.mark.parametrize("cls", [EPR, WEPR, EPRClassifier, WEPRClassifier])
def test_a_detector_or_classifier_without_k_is_refused(cls):
    with pytest.raises(TypeError, match="'k'"):
        cls()  # ty: ignore[missing-argument]


def test_a_pooling_detector_without_k_is_refused():
    with pytest.raises(TypeError, match="'k'"):
        PoolingDetector(EPRPooling())  # ty: ignore[missing-argument]


def test_an_entropy_transformer_without_k_is_refused():
    with pytest.raises(TypeError, match="'k'"):
        EntropyTransformer(pooling=EPRPooling())  # ty: ignore[missing-argument]


def test_an_entropy_transformer_with_k_none_keeps_every_rank():
    cube = -np.random.RandomState(0).exponential(1.0, size=(2, 3, 20))

    assert EntropyTransformer(pooling=WEPRPooling(), k=None).transform(cube).shape == (2, 40)
    assert EntropyTransformer(pooling=WEPRPooling(), k=15).transform(cube).shape == (2, 30)
    assert EntropyTransformer(pooling=EPRPooling(), k=None).validate(cube).shape == cube.shape


@pytest.mark.parametrize("cls", [EPR, WEPR])
@pytest.mark.parametrize("identifier", ["artefactory/does-not-exist", "/no/such/model.skops"])
def test_from_pretrained_without_k_is_refused_before_any_lookup(cls, identifier):
    with pytest.raises(TypeError, match="pass the k the model was trained with"):
        cls.from_pretrained(identifier)


@pytest.mark.parametrize("identifier", ["artefactory/does-not-exist", "/no/such/model.skops"])
def test_pooling_detector_from_pretrained_without_k_is_refused_before_any_lookup(identifier):
    with pytest.raises(TypeError, match="pass the k the model was trained with"):
        PoolingDetector.from_pretrained(identifier, pooling=EPRPooling())


# The file holds no `k`, so the only check is the width the pooling implies. EPR pools to
# one feature at every `k`, so a wrong `k` cannot be told from the right one.
@pytest.mark.parametrize("k", [3, 15, 40])
def test_epr_accepts_weights_at_any_k(k):
    assert EPR.from_estimator(fitted_logistic(0.0, [1.0]), "somewhere", k=k).k == k


def test_wepr_refuses_weights_fitted_at_another_k():
    weights = fitted_logistic(0.0, [1.0] * 30)  # a mean and a max per rank, at k=15

    assert WEPR.from_estimator(weights, "somewhere", k=15).k == 15
    with pytest.raises(ValueError, match="somewhere"):
        WEPR.from_estimator(weights, "somewhere", k=10)


@pytest.mark.parametrize("cls", [EPR, WEPR, EPRClassifier, WEPRClassifier])
def test_from_estimator_without_k_is_refused(cls):
    with pytest.raises(TypeError, match="pass the k the model was trained with"):
        cls.from_estimator(fitted_logistic(0.0, [1.0]), "somewhere")
