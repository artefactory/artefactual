"""How `EntropyTransformer` dispatches its reduction.

Inputs are drawn rather than fixed: the transformer promises to reduce any rectangular
rank cube the parser can emit, and one hand-written array only proves it for that shape.
"""

import numpy as np
import pytest
from conftest import logprob_cubes
from hypothesis import given

from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer


@pytest.mark.parametrize("reduction", ["epr", "wepr"])
@given(logprobs=logprob_cubes())
def test_known_reductions_transform(reduction, logprobs):
    features = EntropyTransformer(reduction=reduction).transform(logprobs)
    assert features.shape[0] == logprobs.shape[0]
    assert np.isfinite(features).all()


@given(logprobs=logprob_cubes())
def test_callable_reduction_is_used_as_is(logprobs):
    features = EntropyTransformer(reduction=lambda x, axis: np.nanmean(x, axis=axis)).transform(logprobs)
    assert features.shape[0] == logprobs.shape[0]


@given(logprobs=logprob_cubes())
def test_unknown_reduction_raises_value_error(logprobs):
    with pytest.raises(ValueError, match="Invalid reduction: 'bogus'"):
        EntropyTransformer(reduction="bogus").transform(logprobs)


@given(logprobs=logprob_cubes())
def test_unknown_reduction_error_names_the_valid_options(logprobs):
    with pytest.raises(ValueError, match="Expected 'epr', 'wepr', or a callable"):
        EntropyTransformer(reduction="mean").transform(logprobs)


@given(logprobs=logprob_cubes())
def test_reduction_parameter_is_not_mutated(logprobs):
    # get_params/set_params/clone require constructor args to survive untouched
    transformer = EntropyTransformer(reduction="wepr")
    transformer.transform(logprobs)
    assert transformer.reduction == "wepr"
    assert transformer.get_params()["reduction"] == "wepr"


# --- input validation --------------------------------------------------------------------


@pytest.mark.parametrize("reduction", ["epr", "wepr"])
@given(logprobs=logprob_cubes())
def test_a_nested_list_scores_like_the_array(reduction, logprobs):
    transformer = EntropyTransformer(reduction=reduction)
    assert np.array_equal(transformer.transform(logprobs.tolist()), transformer.transform(logprobs))
    assert np.array_equal(transformer.transform_tokens(logprobs.tolist()), transformer.transform_tokens(logprobs))


@pytest.mark.parametrize("method", ["transform", "transform_tokens"])
def test_infinite_logprobs_are_refused(method):
    cube = np.full((1, 2, 3), -0.5)
    cube[0, 1, 2] = -np.inf
    with pytest.raises(ValueError, match="infinity"):
        getattr(EntropyTransformer(), method)(cube)


def test_an_empty_batch_passes_validation():
    assert EntropyTransformer().transform(np.empty((0, 0, 0))).shape[0] == 0


@pytest.mark.parametrize(
    ("params", "name"),
    [({"reduction": "bogus"}, "reduction"), ({"k": 0}, "k"), ({"k": 2.5}, "k"), ({"reduction": 3}, "reduction")],
)
def test_fit_rejects_invalid_parameters(params, name):
    with pytest.raises(ValueError, match=f"'{name}'"):
        EntropyTransformer(**params).fit(np.zeros((1, 1, 1)))


def test_fit_accepts_a_callable_reduction_and_no_k():
    assert EntropyTransformer(reduction=lambda x, axis: np.nanmean(x, axis=axis), k=None).fit(None)


def test_fit_with_no_data_records_no_width():
    transformer = EntropyTransformer().fit(None)
    assert "n_features_in_" not in vars(transformer)
