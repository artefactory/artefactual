import numpy as np
import pytest

from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer

# (n_sequences, n_tokens, k) log-probabilities, descending along the rank axis
LOGPROBS = np.array([[[-0.1, -2.0, -3.0], [-0.5, -1.5, -4.0]]])


@pytest.mark.parametrize("reduction", ["epr", "wepr"])
def test_known_reductions_transform(reduction):
    features = EntropyTransformer(reduction=reduction).transform(LOGPROBS)
    assert features.shape[0] == LOGPROBS.shape[0]
    assert np.isfinite(features).all()


def test_callable_reduction_is_used_as_is():
    features = EntropyTransformer(reduction=lambda x, axis: np.nanmean(x, axis=axis)).transform(LOGPROBS)
    assert features.shape[0] == LOGPROBS.shape[0]


def test_unknown_reduction_raises_value_error():
    with pytest.raises(ValueError, match="Invalid reduction: 'bogus'"):
        EntropyTransformer(reduction="bogus").transform(LOGPROBS)


def test_unknown_reduction_error_names_the_valid_options():
    with pytest.raises(ValueError, match="Expected 'epr', 'wepr', or a callable"):
        EntropyTransformer(reduction="mean").transform(LOGPROBS)


def test_reduction_parameter_is_not_mutated():
    # get_params/set_params/clone require constructor args to survive untouched
    transformer = EntropyTransformer(reduction="wepr")
    transformer.transform(LOGPROBS)
    assert transformer.reduction == "wepr"
    assert transformer.get_params()["reduction"] == "wepr"
