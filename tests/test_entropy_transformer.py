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
