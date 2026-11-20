"""The `EPR` / `WEPR` detectors and `PoolingDetector`, which score completion responses."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils import Tags

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_classifier import DEFAULT_K, BaseEntropyClassifier
from artefactual.scoring.entropy_methods.pooling import EPRPooling, Pooling, WEPRPooling, as_pooling


class BaseDetector(BaseEntropyClassifier):
    """A `parser -> entropy -> classifier` pipeline returning P(hallucination) from responses.

    A `BaseEntropyClassifier` with a `LogProbParser` in front, so it takes completion
    responses rather than logprob arrays. It keeps everything the classifier does -- the
    parameters, targets, persistence and `predict_token_proba` -- and adds only the parsing.

    Abstract in the pooling: `EPR` and `WEPR` are the detectors to construct, and
    `PoolingDetector` takes any pooling. A subclass fixes its pooling with the `pooling`
    class attribute, a `Pooling` instance.

    Input is completion responses, one row per sampled sequence: chat completions,
    Responses API payloads or batch output lines. A numeric table is refused with a
    `TypeError`; scoring one is what `EPRClassifier` and `WEPRClassifier` are for.

    Parameters
    ----------
    k : int, default=15
        Candidates per token the responses carry. Responses carrying fewer are rejected
        when parsed, rather than padded, since the missing candidates were never fetched.
        Written to the parser and the entropy step, and any loaded weights were checked
        against it beforehand.
    estimator : estimator, default=None
        Final estimator. Unfitted by default: the unregularised logistic regression the
        published detectors were fit with.
    transform_input : list of str, default=None
        Passed to `Pipeline`.
    memory : str or object with the joblib.Memory interface, default=None
        Passed to `Pipeline`.
    verbose : bool, default=False
        Passed to `Pipeline`.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The class labels, sorted. Available once fitted.
    named_steps : Bunch
        The steps by name: `"parser"`, `"entropy"` and `"classifier"`.
    steps : list of (str, estimator)
        The same steps, in order.
    estimator : estimator
        The final step.

    Raises
    ------
    TypeError
        If constructed directly rather than through `EPR`, `WEPR` or `PoolingDetector`.

    See Also
    --------
    EPR : One pooled feature.
    WEPR : A mean and a max per rank.
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        estimator: BaseEstimator | None = None,
        *,
        transform_input=None,
        memory=None,
        verbose=False,
    ) -> None:
        """Assemble a parser -> entropy -> classifier pipeline pinned to `k` candidates."""
        super().__init__(k=k, estimator=estimator, transform_input=transform_input, memory=memory, verbose=verbose)

    def assemble_steps(self, pooling, k, estimator, transform_input, memory, verbose) -> None:
        """Build the entropy -> classifier steps, then put the parser in front."""
        super().assemble_steps(pooling, k, estimator, transform_input, memory, verbose)
        self.steps.insert(0, ("parser", LogProbParser(k=k)))

    def __sklearn_tags__(self) -> Tags:
        """Declares a binary classifier over responses, which are not an array."""
        # The classifier's tags describe a numeric table, and the front of this pipeline
        # takes none; the parser's own tags are what `Pipeline` reports for its input.
        tags = Pipeline.__sklearn_tags__(self)
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = as_pooling(self.pooling).constant_width
        return tags


class EPR(BaseDetector):
    """A detector that pools a response's uncertainty into one number.

    EPR -- Entropy Production Rate. A single feature, pooling every candidate of the token
    distribution into one number, so the calibration fits one coefficient.

    Input is completion responses. Targets are binary, 1 marking hallucination. To score
    logprob arrays, use `EPRClassifier`.

    Both detectors need weights fit on labelled data, so choosing this one saves no setup
    work over `WEPR` -- only parameters. Prefer `WEPR` unless there is too little labelled
    data to fit its larger coefficient vector.

    Parameters
    ----------
    k : int, default=15
        Candidates per token the responses carry. See `BaseDetector`.
    estimator : estimator, default=None
        Final estimator; the unregularised logistic regression by default.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The class labels, sorted. Available once fitted.
    named_steps : Bunch
        The steps by name: `"parser"`, `"entropy"` and `"classifier"`.
    steps : list of (str, estimator)
        The same steps, in order.
    estimator : estimator
        The final step.

    See Also
    --------
    WEPR : Reads each rank separately, at the cost of `2k` coefficients.
    EPRClassifier : The same reduction over logprob arrays.

    Examples
    --------
    >>> detector = EPR(k=15).fit(responses, y)  # doctest: +SKIP
    >>> published = EPR.from_pretrained("artefactory/epr-phi4", k=15)  # doctest: +SKIP
    >>> published.predict_proba(response)[:, 1]  # doctest: +SKIP
    """

    pooling: ClassVar[Pooling | None] = EPRPooling()


class WEPR(BaseDetector):
    """A detector that reads each rank of the token distribution separately.

    WEPR -- Weighted EPR. `2k` features, a mean and a max per rank, letting the
    calibration weight the informative ranks over the rest -- the second-most-likely
    candidate carries different information from the tenth, and a coefficient per rank is
    what lets the fit say so. It reads strictly more of the distribution than `EPR`, which
    makes it the more accurate of the two when there is enough labelled data to fit `2k`
    coefficients.

    Input is completion responses. Targets are binary, 1 marking hallucination. To score
    logprob arrays, use `WEPRClassifier`.

    Parameters
    ----------
    k : int, default=15
        Candidates per token the responses carry. See `BaseDetector`.
    estimator : estimator, default=None
        Final estimator; the unregularised logistic regression by default.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The class labels, sorted. Available once fitted.
    named_steps : Bunch
        The steps by name: `"parser"`, `"entropy"` and `"classifier"`.
    steps : list of (str, estimator)
        The same steps, in order.
    estimator : estimator
        The final step.

    See Also
    --------
    EPR : One pooled feature, for scarce labelled data.
    WEPRClassifier : The same reduction over logprob arrays.

    Examples
    --------
    >>> detector = WEPR(k=15).fit(responses, y)  # doctest: +SKIP
    >>> published = WEPR.from_pretrained(
    ...     "artefactory/wepr-phi4", k=15
    ... )  # doctest: +SKIP
    >>> published.predict_proba(response)[:, 1]  # doctest: +SKIP
    """

    pooling: ClassVar[Pooling | None] = WEPRPooling()


class PoolingDetector(BaseDetector):
    """A detector over any pooling: a `Pooling` instance or a function `f(x, axis)`.

    The pooling is a constructor parameter, so `get_params`, `clone` and a grid search see
    it, and it is never a class attribute. Everything else is `BaseDetector`.

    The saved file holds the fitted classifier alone, not the pooling, so loading one with
    `from_pretrained` needs the pooling it was fitted with: pass `pooling=` there.

    Parameters
    ----------
    pooling : Pooling or callable
        A `Pooling` instance, or a function `f(x, axis) -> (..., n_features)` pooling the
        token `axis` of NaN-padded entropy contributions whose last axis is the rank axis.
    k : int, default=15
        Candidates per token the responses carry. See `BaseDetector`.
    estimator : estimator, default=None
        Final estimator; the unregularised logistic regression by default.
    transform_input : list of str, default=None
        Passed to `Pipeline`.
    memory : str or object with the joblib.Memory interface, default=None
        Passed to `Pipeline`.
    verbose : bool, default=False
        Passed to `Pipeline`.

    Attributes
    ----------
    pooling : Pooling or callable
        Read from, and writing it replaces, the pooling of the entropy step.
    classes_ : ndarray of shape (2,)
        The class labels, sorted. Available once fitted.
    named_steps : Bunch
        The steps by name: `"parser"`, `"entropy"` and `"classifier"`.

    Raises
    ------
    TypeError
        When fitted or scored, if `pooling` is neither a `Pooling` nor callable.

    See Also
    --------
    Pooling : The base class to subclass.
    EPR : The detector for the shipped `EPRPooling`.

    Examples
    --------
    >>> class Median(Pooling):
    ...     def __call__(self, X, axis):
    ...         return np.nanmedian(np.nansum(X, axis=-1, keepdims=True), axis=axis)
    >>> detector = PoolingDetector(Median(), k=15).fit(responses, y)  # doctest: +SKIP
    """

    def __init__(
        self,
        pooling: Pooling | Callable,
        k: int = DEFAULT_K,
        estimator: BaseEstimator | None = None,
        *,
        transform_input=None,
        memory=None,
        verbose=False,
    ) -> None:
        """Assemble a parser -> entropy -> classifier pipeline over `pooling`."""
        self.assemble_steps(pooling, k, estimator, transform_input, memory, verbose)

    @classmethod
    def from_pretrained(  # ty: ignore[incompatible-method-override]
        cls, identifier: str | Path, *, pooling: Pooling | Callable, **kwargs: Any
    ) -> "PoolingDetector":
        """Load the fitted classifier at `identifier` and score with `pooling`.

        The file holds the classifier alone, so `pooling` must be the one it was fitted with;
        it is required because no pooling is a safe guess. Other options are those of
        `PretrainedMixin.from_pretrained`, and `k` the width it was fitted at.
        """
        return super().from_pretrained(identifier, pooling=pooling, **kwargs)

    @property
    def pooling(self) -> Pooling | Callable:
        """The pooling, read off the entropy step so `set_params(pooling=)` and `k` agree on one source."""
        return self.named_steps["entropy"].pooling

    @pooling.setter
    def pooling(self, value: Pooling | Callable) -> None:  # ty: ignore[incompatible-property-override]
        self.named_steps["entropy"].pooling = value
