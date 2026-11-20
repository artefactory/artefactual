"""Poolings: the objects that turn entropy contributions into the features a calibration reads.

A pooling is a `Pooling` instance, or a plain function `f(x, axis) -> (..., n_features)`.
`x` holds NaN-padded entropy contributions whose last axis is the rank axis, and `axis` is
the axis to pool over: the token axis. Subclass `Pooling` to name the features, declare the
width or carry parameters; pass a bare function when its name and a probed width are enough.
There is no registry: a pooling is always passed as an object.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from beartype.door import is_bearable

# Probed at a few widths to tell a width that depends on `k` from one that does not.
_PROBE_WIDTHS = (1, 2, 3)


def epr_reduction(x: np.ndarray, axis: int) -> np.ndarray:
    """Entropy Production Rate: the truncated entropy, averaged over tokens.

    The paper's Eq. 6 -- the mean over the sequence of the top-K entropy estimate
    `H~_K = -sum_k p_k ln p_k` (Eq. 3). So the rank axis is summed, not averaged: the
    result is an entropy, on the same scale as `H(q)` itself, in nats as the ECIR2026
    release computes it.

    A token whose candidates are entirely NaN is padding rather than data, and is excluded from
    the token mean rather than contributing the 0 that `nansum` would give it.

    A fully padded sequence is an expected input, not an anomaly: `EntropyTransformer.transform`
    detects it and raises `EmptySequenceWarning`, which names the condition. numpy's "Mean of
    empty slice" reports the same thing a second time from a frame that points here, so it is
    silenced inside this reduction only, and a caller-supplied reduction keeps its own warnings.

    Parameters
    ----------
    x : ndarray of shape (..., n_tokens, k)
        Entropy contributions, NaN-padded; the last axis is the rank axis.
    axis : int
        The token axis to average over.

    Returns
    -------
    ndarray
        Shape `(n_sequences, 1)`: one value per sequence, NaN where the sequence has no token.
    """
    padded = np.all(np.isnan(x), axis=-1, keepdims=True)  # fully-NaN (padded) tokens
    s = np.nansum(x, axis=-1, keepdims=True)  # H~_K: sum over the K ranks
    s = np.where(padded, np.nan, s)  # nansum gave 0 for padded tokens → restore NaN
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        return np.nanmean(s, axis=axis)  # EPR: mean over the token axis


def wepr_reduction(x: np.ndarray, axis: int) -> np.ndarray:
    """Weighted EPR features: the per-rank mean over tokens, then the per-rank max.

    Parameters
    ----------
    x : ndarray of shape (..., n_tokens, k)
        Entropy contributions, NaN-padded; the last axis is the rank axis.
    axis : int
        The token axis to reduce.

    Returns
    -------
    ndarray
        The mean branch then the max branch concatenated on the last axis, so `2k` features
        per sequence, NaN where the sequence has no token. The "Mean of empty slice" and
        "All-NaN slice encountered" warnings are silenced inside this reduction, for the
        reason `epr_reduction` gives.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        mean_branch = np.nanmean(x, axis=axis)
        max_branch = np.nanmax(x, axis=axis)
    return np.concatenate([mean_branch, max_branch], axis=-1)


# A module-level helper, not a method: it recurses into values that are not poolings.
def _same_value(a: Any, b: Any) -> bool:
    """Whether two attribute values are the same object or hold the same value, arrays included."""
    if a is b:
        return True
    if is_bearable(a, np.ndarray) and is_bearable(b, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
    if type(a) is type(b) and type(a) in {list, tuple}:
        return len(a) == len(b) and all(map(_same_value, a, b))
    if type(a) is type(b) and type(a) is dict:
        return a.keys() == b.keys() and all(_same_value(a[key], b[key]) for key in a)
    try:
        return bool(a == b) or bool(a != a and b != b)  # NaN is not equal to itself
    except (TypeError, ValueError):  # an `==` with no single truth value, as a DataFrame's
        return False


class Pooling:
    """Pools the token axis of NaN-padded entropy contributions; the last axis is ranks.

    Subclass it and implement `__call__`. The class name, lower-cased, is the pooling's
    `name`, which names a one-feature result and prefixes a wider one (`name_0`, `name_1`,
    ...). Override `name` to rename it, `feature_names` to name a result after the rank
    columns, `width` to declare the feature count instead of probing it, and set
    `constant_width = True` when the count does not depend on `k`.

    Instances compare equal when they have the same type and the same attributes, so a
    `sklearn.base.clone` of an estimator holding one compares equal to the original. Arrays,
    alone or inside a list, a tuple or a dict, are compared by value, NaN equal to NaN; any
    other attribute has to be the same object or equal under `==`.

    Examples
    --------
    >>> class Median(Pooling):
    ...     def __call__(self, X, axis):
    ...         return np.nanmedian(np.nansum(X, axis=-1, keepdims=True), axis=axis)
    >>> PoolingDetector(Median(), k=15)  # doctest: +SKIP
    """

    @property
    def name(self) -> str:
        """The feature name of a one-feature pooling and the prefix of a wider one."""
        return type(self).__name__.lower()

    def __call__(self, X: np.ndarray, axis: int) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Pool `X` over `axis`.

        Parameters
        ----------
        X : ndarray of shape (..., n_tokens, k)
            Entropy contributions, NaN-padded; the last axis is the rank axis.
        axis : int
            The token axis to pool.

        Returns
        -------
        ndarray of shape (..., n_features)
            The `axis` replaced by nothing and the rank axis by the features.
        """
        raise NotImplementedError

    def width(self, k: int) -> int:
        """Features the pooling returns for `k` candidates per token.

        Read off the output of the pooling on a zeros array of shape `(1, 2, k)` pooled over
        axis 1, the axis `EntropyTransformer.transform` pools. Override it to declare the
        count instead; a pooling that fails on zeros must.

        Raises
        ------
        ValueError
            If the pooling fails on that probe or does not return `(n_sequences, n_features)`.
        """
        probe = np.zeros((1, 2, k))
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                out = np.asarray(self(probe, 1))
            except Exception as error:
                msg = (
                    f"Pooling {self.name!r} failed on a zeros array of shape (1, 2, {k}) pooled over axis=1, "
                    f"which is how its feature count is read; override `width` to declare it: {error}"
                )
                raise ValueError(msg) from error
        if out.ndim != 2:
            msg = (
                f"Pooling {self.name!r} returned shape {out.shape} for an input of shape (1, 2, {k}) "
                "pooled over axis=1; expected (n_sequences, n_features)."
            )
            raise ValueError(msg)
        return out.shape[-1]

    @property
    def constant_width(self) -> bool:
        """Whether the feature count is the same whatever `k` is, as EPR's single feature is.

        Probed at `k` of 1, 2 and 3, so a width that changes only beyond that is missed:
        override it with a bool to declare it.
        """
        return len({self.width(k) for k in _PROBE_WIDTHS}) == 1

    def feature_names(self, ranks: Sequence[str]) -> list[str]:
        """One name per feature, given the rank column names.

        The default is `[name]` for one feature and `name_0`, `name_1`, ... for more. A
        constant-width pooling may be asked with a placeholder rank when `k` is unknown.
        """
        width = self.width(len(ranks))
        return [self.name] if width == 1 else [f"{self.name}_{i}" for i in range(width)]

    def __eq__(self, other: Any) -> bool:
        return type(self) is type(other) and _same_value(vars(self), vars(other))

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        arguments = ", ".join(f"{key}={value!r}" for key, value in vars(self).items())
        return f"{type(self).__name__}({arguments})"


class FunctionPooling(Pooling):
    """A plain function `f(x, axis)` as a `Pooling`, named after the function.

    A lambda or a `functools.partial` has no name: pass `name=`.

    Parameters
    ----------
    fn : callable
        `fn(x, axis)` pooling the `axis` of `x`, whose last axis is the rank axis, to
        `(..., n_features)`.
    name : str, default=None
        The pooling's name. Defaults to `fn.__name__`, else `"pooling"`.
    """

    def __init__(self, fn: Callable, name: str | None = None) -> None:
        self.fn = fn
        self.label = name if name is not None else _name_of(fn)

    @property
    def name(self) -> str:
        return self.label

    def __call__(self, X: np.ndarray, axis: int) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Pool `X` over `axis` with the wrapped function."""
        return self.fn(X, axis)

    def __repr__(self) -> str:
        return f"FunctionPooling({self.label})"


# A module-level helper, not a method: only `FunctionPooling.__init__` needs it.
def _name_of(fn: Callable) -> str:
    """The `__name__` of `fn` when it identifies it, `"pooling"` for a lambda or a `partial`."""
    try:
        name = fn.__name__  # ty: ignore[missing-attribute]
    except AttributeError:
        return "pooling"
    return "pooling" if name == "<lambda>" else name


class EPRPooling(Pooling):
    """The Entropy Production Rate: one feature, `epr_reduction`."""

    constant_width = True

    @property
    def name(self) -> str:
        return "epr"

    def __call__(self, X: np.ndarray, axis: int) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """`epr_reduction` of `X` over `axis`: shape `(n_sequences, 1)`."""
        return epr_reduction(X, axis)

    def width(self, k: int) -> int:  # ruff: ignore[unused-method-argument]
        """One feature, whatever `k` is."""
        return 1

    def feature_names(self, ranks: Sequence[str]) -> list[str]:  # ruff: ignore[unused-method-argument]
        """`["epr"]`: every rank is pooled into the one feature."""
        return ["epr"]


class WEPRPooling(Pooling):
    """The Weighted EPR: a mean and a max per rank, `wepr_reduction`."""

    constant_width = False

    @property
    def name(self) -> str:
        return "wepr"

    def __call__(self, X: np.ndarray, axis: int) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """`wepr_reduction` of `X` over `axis`: shape `(n_sequences, 2k)`."""
        return wepr_reduction(X, axis)

    def width(self, k: int) -> int:
        """`2k`: the mean branch then the max branch."""
        return 2 * k

    def feature_names(self, ranks: Sequence[str]) -> list[str]:
        """`wepr_mean_<rank>` for each rank, then `wepr_max_<rank>` for each rank."""
        return [f"wepr_{branch}_{rank}" for branch in ("mean", "max") for rank in ranks]


def as_pooling(pooling: Pooling | Callable | None) -> Pooling:
    """`pooling` as a `Pooling`: an instance is returned as it is, a function is wrapped.

    Raises
    ------
    TypeError
        If `pooling` is neither a `Pooling` nor callable (`None` and a string included: poolings are
        passed as objects, not looked up by name).
    """
    if is_bearable(pooling, Pooling):
        return pooling
    if callable(pooling):
        return FunctionPooling(pooling)
    msg = (
        f"Invalid pooling {pooling!r}: pass a Pooling instance (EPRPooling(), WEPRPooling(), "
        "or a subclass of Pooling) or a function f(x, axis). Poolings are not selected by name."
    )
    raise TypeError(msg)
