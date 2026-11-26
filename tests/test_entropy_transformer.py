"""How `EntropyTransformer` dispatches its reduction.

Inputs are drawn: the transformer reduces any rectangular rank cube the parser can emit.
"""

import numpy as np
import pytest
from conftest import logprob_cubes
from hypothesis import given

from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.entropy_methods.pooling import EPRPooling, WEPRPooling


@pytest.mark.parametrize("pooling", [EPRPooling(), WEPRPooling()], ids=["epr", "wepr"])
@given(logprobs=logprob_cubes())
def test_known_poolings_transform(pooling, logprobs):
    features = EntropyTransformer(pooling=pooling).transform(logprobs)
    assert features.shape[0] == logprobs.shape[0]
    assert np.isfinite(features).all()


@given(logprobs=logprob_cubes())
def test_callable_pooling_is_used_as_is(logprobs):
    features = EntropyTransformer(pooling=lambda x, axis: np.nanmean(x, axis=axis)).transform(logprobs)
    assert features.shape[0] == logprobs.shape[0]


@pytest.mark.parametrize("name", ["epr", "wepr", "bogus"])
@given(logprobs=logprob_cubes())
def test_a_string_pooling_is_refused_and_says_what_to_pass(name, logprobs):
    transformer = EntropyTransformer(pooling=name)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match="Pooling instance"):
        transformer.transform(logprobs)
    with pytest.raises(TypeError, match="Pooling instance"):
        transformer.get_feature_names_out()


@given(logprobs=logprob_cubes())
def test_pooling_parameter_is_not_mutated(logprobs):
    # get_params/set_params/clone require constructor args to survive untouched
    pooling = WEPRPooling()
    transformer = EntropyTransformer(pooling=pooling)
    transformer.transform(logprobs)
    assert transformer.pooling is pooling
    assert transformer.get_params()["pooling"] is pooling


# --- input validation --------------------------------------------------------------------


@pytest.mark.parametrize("pooling", [EPRPooling(), WEPRPooling()], ids=["epr", "wepr"])
@given(logprobs=logprob_cubes())
def test_a_nested_list_scores_like_the_array(pooling, logprobs):
    transformer = EntropyTransformer(pooling=pooling)
    assert np.array_equal(transformer.transform(logprobs.tolist()), transformer.transform(logprobs))
    assert np.array_equal(transformer.transform_tokens(logprobs.tolist()), transformer.transform_tokens(logprobs))


@pytest.mark.parametrize("method", ["transform", "transform_tokens"])
def test_infinite_logprobs_are_refused(method):
    cube = np.full((1, 2, 3), -0.5)
    cube[0, 1, 2] = -np.inf
    with pytest.raises(ValueError, match="infinity"):
        getattr(EntropyTransformer(pooling=EPRPooling()), method)(cube)


def test_an_empty_batch_passes_validation():
    assert EntropyTransformer(pooling=EPRPooling()).transform(np.empty((0, 0, 0))).shape[0] == 0


def test_fit_rejects_an_invalid_pooling():
    with pytest.raises(ValueError, match="'pooling'"):
        EntropyTransformer(pooling=3).fit(np.zeros((1, 1, 1)))  # ty: ignore[invalid-argument-type]


def test_fit_accepts_a_callable_pooling():
    assert EntropyTransformer(pooling=lambda x, axis: np.nanmean(x, axis=axis)).fit(None)


def test_fit_with_no_data_records_no_width():
    transformer = EntropyTransformer(pooling=EPRPooling()).fit(None)
    assert "n_features_in_" not in vars(transformer)
