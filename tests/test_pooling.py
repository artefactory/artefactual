"""Poolings: a `Pooling` subclass or any `f(x, axis)` is a feature extractor, passed as an object.

Real arrays and real fits throughout; there is no registry, so no state is shared between tests.
"""

import copy
import dataclasses
import functools
import inspect
from typing import Any, ClassVar

import numpy as np
import pytest
import sklearn.base
from conftest import logprob_cubes
from hypothesis import given

import artefactual
from artefactual.scoring.entropy_methods.entropy_contributions import EntropyContributionsMixin

K = 3
CUBE = np.array([
    [[-0.1, -2.0, -3.0], [-0.5, -1.5, -4.0], [np.nan, np.nan, np.nan]],
    [[-0.3, -1.0, -2.5], [-0.2, -2.2, -3.1], [-0.9, -1.1, -1.3]],
])
Y = np.array([0, 1])


def _chat(ranks, n_tokens=2):
    token = {"token": "t", "logprob": ranks[0], "top_logprobs": [{"logprob": r} for r in ranks]}
    return {"choices": [{"logprobs": {"content": [token] * n_tokens}}]}


RESPONSES = [
    _chat([-0.001, -8.0, -9.0]),
    _chat([-1.0, -1.1, -1.2]),
    _chat([-0.01, -6.0, -7.0]),
    _chat([-0.9, -1.0, -1.4]),
]
LABELS = np.array([0, 1, 0, 1])


def module_median(x, axis):
    """Module level, so that a pickle can name it."""
    per_token = np.nansum(x, axis=-1, keepdims=True)
    return np.nanmedian(per_token, axis=axis)


def per_rank_median(x, axis):
    return np.nanmedian(x, axis=axis)


@dataclasses.dataclass(frozen=True)
class Median(artefactual.Pooling):
    """One feature: the median over tokens of the entropy summed over ranks."""

    constant_width: ClassVar[bool] = True

    def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
        return module_median(X, axis)


@dataclasses.dataclass(frozen=True)
class RankMedian(artefactual.Pooling):
    """One median per rank, named after the rank columns, with a parameter."""

    scale: Any = 1.0

    @property
    def name(self):
        return "rank_median"

    def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
        return self.scale * np.nanmedian(X, axis=axis)

    def feature_names(self, ranks):
        return [f"med_{r}" for r in ranks]


def _expected_median(cube):
    contributions = EntropyContributionsMixin().entropy_contributions(cube)
    return module_median(contributions, axis=1)


# --- a function is a pooling ------------------------------------------------------------


def test_a_function_pools_and_is_named_after_itself():
    transformer = artefactual.EntropyTransformer(pooling=module_median)

    assert list(transformer.get_feature_names_out()) == ["module_median"]
    assert transformer.transform(CUBE).shape == (2, 1)


@given(cube=logprob_cubes())
def test_a_function_pooling_equals_applying_it_to_the_contributions(cube):
    features = artefactual.EntropyTransformer(pooling=module_median).transform(cube)

    np.testing.assert_array_equal(features, _expected_median(cube))


def test_a_function_pooling_has_a_token_mode():
    tokens = artefactual.EntropyTransformer(pooling=per_rank_median).transform_tokens(CUBE)

    assert tokens.shape == (2, 3, K)


def test_a_function_pooling_is_never_given_a_padding_token():
    tokens = artefactual.EntropyTransformer(pooling=per_rank_median).transform_tokens(CUBE)

    assert np.isnan(tokens[0, 2]).all()
    assert not np.isnan(tokens[:, :2]).any()
    assert not np.isnan(tokens[1]).any()


def test_a_function_pooling_has_a_token_mode_without_any_real_token():
    tokens = artefactual.EntropyTransformer(pooling=per_rank_median).transform_tokens(np.full((2, 3, K), np.nan))

    assert np.isnan(tokens).all()


def test_a_wide_function_names_its_features_by_position():
    names = artefactual.EntropyTransformer(pooling=per_rank_median).fit(CUBE).get_feature_names_out()

    assert list(names) == ["per_rank_median_0", "per_rank_median_1", "per_rank_median_2"]


def test_an_unnamed_width_is_refused_until_the_step_is_fitted():
    with pytest.raises(ValueError, match="unknown"):
        artefactual.EntropyTransformer(pooling=per_rank_median).get_feature_names_out()


def test_a_lambda_or_partial_function_pooling_is_named_pooling():
    pooled = functools.partial(np.nanmean, keepdims=False)

    assert artefactual.FunctionPooling(lambda x, axis: np.nanmean(x, axis=axis)).name == "pooling"
    assert artefactual.FunctionPooling(pooled).name == "pooling"


def test_a_pooling_subclass_that_is_not_a_dataclass_is_refused():
    class Plain(artefactual.Pooling):
        def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
            return np.nanmean(X, axis=axis)

    with pytest.raises(TypeError, match=r"Plain.*@dataclasses\.dataclass\(frozen=True\)"):
        Plain()


def test_a_pooling_of_the_wrong_type_is_refused():
    with pytest.raises(ValueError, match="pooling"):
        artefactual.EntropyTransformer(pooling=3).fit(CUBE)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Pooling instance"):
        artefactual.EntropyTransformer(pooling=3).transform(CUBE)  # ty: ignore[invalid-argument-type]


# --- a Pooling subclass -------------------------------------------------------------------


def test_a_subclass_is_named_after_its_class_and_declares_nothing_it_does_not_need():
    pooling = Median()

    assert pooling.name == "median"
    assert pooling.width(K) == 1
    assert pooling.feature_names(["a", "b", "c"]) == ["median"]
    assert pooling.constant_width is True


def test_the_default_width_is_probed_and_names_follow_it():
    @dataclasses.dataclass(frozen=True)
    class Mean(artefactual.Pooling):
        def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
            return np.nanmean(X, axis=axis)

    pooling = Mean()

    assert pooling.width(4) == 4
    assert pooling.feature_names(["a", "b"]) == ["mean_0", "mean_1"]
    assert pooling.constant_width is False


def test_a_probe_that_fails_is_reported_with_the_way_out():
    @dataclasses.dataclass(frozen=True)
    class Fussy(artefactual.Pooling):
        def __call__(self, _x, _axis):
            msg = "no zeros please"
            raise RuntimeError(msg)

    with pytest.raises(ValueError, match="override `width`"):
        Fussy().width(K)


def test_a_pooling_that_declares_its_width_is_never_probed():
    @dataclasses.dataclass(frozen=True)
    class Declared(artefactual.Pooling):
        def __call__(self, _x, _axis):
            msg = "must not be called"
            raise RuntimeError(msg)

        def width(self, _k):
            return 2

        constant_width: ClassVar[bool] = True

    assert Declared().feature_names(["a"]) == ["declared_0", "declared_1"]
    assert list(artefactual.EntropyTransformer(pooling=Declared()).get_feature_names_out()) == [
        "declared_0",
        "declared_1",
    ]


def test_a_subclass_pools_like_the_function_it_wraps():
    np.testing.assert_array_equal(
        artefactual.EntropyTransformer(pooling=Median()).transform(CUBE),
        artefactual.EntropyTransformer(pooling=module_median).transform(CUBE),
    )
    assert list(artefactual.EntropyTransformer(pooling=Median()).get_feature_names_out()) == ["median"]


def test_a_subclass_names_its_features_after_the_rank_columns():
    transformer = artefactual.EntropyTransformer(pooling=RankMedian()).fit(CUBE)

    assert list(transformer.get_feature_names_out()) == ["med_rank0", "med_rank1", "med_rank2"]
    assert list(transformer.get_feature_names_out(["a", "b", "c"])) == ["med_a", "med_b", "med_c"]


def test_feature_names_must_match_the_width():
    @dataclasses.dataclass(frozen=True)
    class Bad(artefactual.Pooling):
        def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
            return np.nanmedian(X, axis=axis)

        def feature_names(self, _ranks):
            return ["only_one"]

    with pytest.raises(ValueError, match="bad"):
        artefactual.EntropyTransformer(pooling=Bad()).fit(CUBE).get_feature_names_out()


def test_the_base_pooling_is_abstract():
    with pytest.raises(NotImplementedError):
        artefactual.Pooling()(CUBE, axis=1)


# --- repr and equality --------------------------------------------------------------------


def test_the_repr_shows_the_class_and_its_parameters():
    assert repr(Median()) == "Median()"
    assert repr(RankMedian(scale=2.0)) == "RankMedian(scale=2.0)"
    assert repr(artefactual.FunctionPooling(module_median)) == (f"FunctionPooling(fn={module_median!r})")
    assert repr(artefactual.EPRPooling()) == "EPRPooling()"


def test_equal_poolings_have_the_same_type_and_parameters():
    assert RankMedian(2.0) == RankMedian(2.0)
    assert RankMedian(2.0) != RankMedian(3.0)
    assert Median() != RankMedian()
    assert artefactual.FunctionPooling(module_median) == artefactual.FunctionPooling(module_median)
    assert artefactual.FunctionPooling(module_median) != artefactual.FunctionPooling(per_rank_median)
    assert hash(artefactual.FunctionPooling(module_median)) == hash(artefactual.FunctionPooling(module_median))
    assert hash(RankMedian(2.0)) == hash(RankMedian(2.0))


def test_an_array_field_declared_compare_false_is_left_out_of_equality_and_hash():
    @dataclasses.dataclass(frozen=True)
    class Plain(artefactual.Pooling):
        weights: Any = dataclasses.field(compare=False)
        tag: str

        def __call__(self, X, axis):  # ruff: ignore[invalid-argument-name]
            return np.nanmedian(X, axis=axis)

    first, second = Plain(np.ones(2), "a"), Plain(np.zeros(2), "a")
    assert first == second
    assert hash(first) == hash(second)
    assert first != Plain(np.ones(2), "b")
    assert "tag='a'" in repr(second)


@pytest.mark.parametrize(
    "pooling",
    [artefactual.EPRPooling(), artefactual.WEPRPooling(), artefactual.FunctionPooling(module_median), RankMedian(2.0)],
    ids=["epr", "wepr", "function", "user"],
)
def test_a_transformer_holding_a_pooling_can_be_cloned(pooling):
    transformer = artefactual.EntropyTransformer(pooling=pooling)
    cloned = sklearn.base.clone(transformer)

    assert cloned is not transformer
    assert cloned.pooling is not pooling
    assert cloned.pooling == pooling
    assert np.array_equal(cloned.fit(CUBE).transform(CUBE), transformer.fit(CUBE).transform(CUBE), equal_nan=True)


# --- the shipped poolings are unchanged -------------------------------------------------


def test_the_shipped_feature_names_are_unchanged():
    epr = artefactual.EntropyTransformer(pooling=artefactual.EPRPooling()).fit(CUBE).get_feature_names_out()
    wepr = artefactual.EntropyTransformer(pooling=artefactual.WEPRPooling()).fit(CUBE).get_feature_names_out()

    assert list(epr) == ["epr"]
    assert list(wepr) == [f"wepr_{b}_rank{i}" for b in ("mean", "max") for i in range(K)]


def test_epr_names_need_no_fit():
    assert list(artefactual.EntropyTransformer(pooling=artefactual.EPRPooling()).get_feature_names_out()) == ["epr"]


def test_a_transformer_without_a_pooling_is_refused():
    with pytest.raises(TypeError, match="pooling"):
        artefactual.EntropyTransformer()  # ty: ignore[missing-argument]


@given(cube=logprob_cubes())
def test_the_shipped_poolings_apply_the_published_reductions(cube):
    contributions = EntropyContributionsMixin().entropy_contributions(cube)

    np.testing.assert_array_equal(
        artefactual.EntropyTransformer(pooling=artefactual.EPRPooling()).transform(cube),
        artefactual.epr_reduction(contributions, axis=1),
    )
    np.testing.assert_array_equal(
        artefactual.EntropyTransformer(pooling=artefactual.WEPRPooling()).transform(cube),
        artefactual.wepr_reduction(contributions, axis=1),
    )


@pytest.mark.parametrize("pooling", [artefactual.EPRPooling(), artefactual.WEPRPooling()], ids=["epr", "wepr"])
@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_a_declared_shipped_width_agrees_with_the_probe(pooling, k):
    probed = artefactual.Pooling.width(pooling, k)

    assert pooling.width(k) == probed
    assert len(pooling.feature_names([f"rank{i}" for i in range(k)])) == probed


def test_the_shipped_constant_width_agrees_with_the_probe():
    for pooling in (artefactual.EPRPooling(), artefactual.WEPRPooling()):
        assert pooling.constant_width is artefactual.Pooling.constant_width.fget(pooling)


def test_the_old_vocabulary_is_gone():
    import artefactual.scoring.entropy_methods.entropy_transformer as module
    import artefactual.utils.io as io

    assert "STRATEGIES" not in dir(module)
    assert "Reduction" not in dir(io)
    assert "reduction" not in artefactual.EntropyTransformer(pooling=artefactual.EPRPooling()).get_params()
    assert "pooling" in artefactual.EntropyTransformer(pooling=artefactual.EPRPooling()).get_params()


# --- estimators -------------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "params"), [(artefactual.EPR, {"estimator", "k"}), (artefactual.WEPR, {"estimator"})])
def test_the_classifiers_keep_their_constructor_surface(cls, params):
    assert list(inspect.signature(cls).parameters) == sorted(params, key=["estimator", "k"].index)
    assert set(cls().get_params(deep=False)) == params
    assert "pooling" not in cls().get_params()


@pytest.mark.parametrize(
    ("cls", "pooling"),
    [
        (artefactual.EPR, artefactual.EPRPooling()),
        (artefactual.WEPR, artefactual.WEPRPooling()),
    ],
)
def test_the_shipped_classifiers_fix_their_pooling(cls, pooling):
    assert cls().fit(CUBE, Y).pipeline_[0].pooling == pooling


def test_a_transformer_stores_its_pooling_untouched():
    pooling = RankMedian()
    by_instance = artefactual.EntropyTransformer(pooling=pooling)
    by_function = artefactual.EntropyTransformer(pooling=module_median)
    by_instance.transform(CUBE)
    by_function.transform(CUBE)

    assert by_instance.pooling is pooling
    assert by_function.pooling is module_median
    assert copy.deepcopy(by_instance).pooling == pooling


# --- widths -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "k", "width"),
    [
        (artefactual.EPR, 15, 1),
        (artefactual.WEPR, 15, 30),
        (artefactual.WEPR, 4, 8),
    ],
)
def test_the_pipeline_width_follows_the_pooling_and_the_fitted_k(cls, k, width):
    cube = -np.random.RandomState(0).exponential(1.0, size=(4, 2, k))

    fitted = cls().fit(cube, LABELS)

    assert len(fitted.get_feature_names_out()) == width


# --- persistence ------------------------------------------------------------------------


def renamed_median(values, dim):
    """The axis parameter has another name: a pooling is called positionally."""
    return np.nanmedian(np.nansum(values, axis=-1, keepdims=True), axis=dim)


@dataclasses.dataclass(frozen=True)
class RenamedMedian(artefactual.Pooling):
    constant_width: ClassVar[bool] = True

    def __call__(self, values, dim):
        return renamed_median(values, dim)


@pytest.mark.parametrize("pooling", [renamed_median, RenamedMedian()], ids=["function", "subclass"])
def test_a_pooling_is_called_positionally_whatever_its_axis_parameter_is_named(pooling):
    expected = artefactual.EntropyTransformer(pooling=module_median).transform(CUBE)

    features = artefactual.EntropyTransformer(pooling=pooling).transform(CUBE)

    np.testing.assert_array_equal(features, expected)
    assert features.shape == (len(CUBE), 1)


@pytest.mark.parametrize(
    "pooling",
    [artefactual.Pooling, artefactual.FunctionPooling, artefactual.EPRPooling, artefactual.WEPRPooling],
    ids=lambda cls: cls.__name__,
)
def test_a_pooling_takes_its_input_as_capital_x(pooling):
    """`X` is the array parameter's name across the package, so keyword callers need one spelling."""
    assert list(inspect.signature(pooling.__call__).parameters) == ["self", "X", "axis"]
