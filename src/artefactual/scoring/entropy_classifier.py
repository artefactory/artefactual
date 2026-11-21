"""The `entropy -> classifier` pipelines over logprob arrays, which every detector builds on."""

from numbers import Integral
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from beartype.door import is_bearable
from sklearn.base import BaseEstimator, _fit_context, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import Tags
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, column_or_1d

from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.entropy_methods.pooling import EPRPooling, Pooling, WEPRPooling, as_pooling
from artefactual.utils.io import PretrainedMixin, SavePretrainedMixin, _require_k


def default_classifier() -> LogisticRegression:
    """The unregularised logistic regression the published detectors were fit with."""
    return LogisticRegression(C=np.inf, max_iter=1000)


class BaseEntropyClassifier(Pipeline, PretrainedMixin, SavePretrainedMixin):
    """An `entropy -> classifier` pipeline returning P(hallucination) from logprobs.

    A scikit-learn `Pipeline`, so `predict`, `predict_proba`, `fit`, `get_params` and
    `clone` behave as expected and the estimator composes into `GridSearchCV` and friends.
    It holds the logic every detector shares: the parameters, the target checks, the
    persistence and the per-token scores. Abstract in the pooling: `EPRClassifier` and
    `WEPRClassifier` are the classes to construct, and each fixes `pooling` and the
    coefficient width that pooling implies.

    Class 1 is the hallucination class: `predict_proba(...)[:, 1]` is the score of
    interest, as long as the labels are `0`/`1`. Column order follows `classes_`.

    Input is the `(n_sequences, n_tokens, width)` logprob array `LogProbParser` emits, or
    a numeric 2-D `(n_samples, width)` table, whose rows are read as one-token sequences
    of per-rank logprobs. Ranks beyond `k` are dropped, which are the least likely
    candidates; a narrower input is used as it is and fits coefficients of that width.
    Targets are binary only.

    `fit` leaves `steps` as the same list object it received, and `get_params` reports
    `estimator=None` while the final step is still the untouched default, so that `clone`
    rebuilds an equal estimator.

    Parameters
    ----------
    k : int
        Candidates per token. Written to every step before the classifier, so that
        `set_params(k=)`, which is how `GridSearchCV` sweeps it, reaches them all.
    estimator : estimator, default=None
        Final estimator. Unfitted by default: the unregularised logistic regression the
        published detectors were fit with, so coefficients fitted here are comparable to
        the shipped ones. `C=np.inf` rather than `penalty=None`: the latter is deprecated
        in scikit-learn 1.8 and removed in 1.10, and the two produce identical
        coefficients.
    transform_input : list of str, default=None
        Passed to `Pipeline`.
    memory : str or object with the joblib.Memory interface, default=None
        Passed to `Pipeline`.
    verbose : bool, default=False
        Passed to `Pipeline`.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The class labels, sorted; column `i` of `predict_proba` is class `classes_[i]`.
        Available once fitted.
    n_features_in_ : int
        Width of the table, or last dimension of the 3-D array, fitted on. Reading it
        before fitting raises `AttributeError`.
    feature_names_in_ : ndarray of str
        Column names of a table fitted on, when they are all strings.
    named_steps : Bunch
        The steps by name: `"entropy"` and `"classifier"`.
    steps : list of (str, estimator)
        The same steps, in order.
    estimator : estimator
        The final step. Read from, and writing it replaces, `steps[-1]`. Writing `None`
        restores the default.

    Raises
    ------
    TypeError
        If constructed directly rather than through `EPRClassifier` or `WEPRClassifier`.

    See Also
    --------
    EPRClassifier : One pooled feature.
    WEPRClassifier : A mean and a max per rank.
    """

    #: The pooling this estimator scores with. Set by each subclass.
    pooling: ClassVar[Pooling | None] = None

    # `k` and `estimator` are properties over the steps, not stored attributes, but
    # `get_params` reports them, so they are validated like any other constructor parameter.
    # `estimator=None` stands for the default classifier.
    # Declared as `Pipeline` declares it, a plain attribute read from the class, so a type
    # checker does not see a ClassVar overriding an instance variable.
    _parameter_constraints: dict = {  # ruff: ignore[mutable-class-default]
        **Pipeline._parameter_constraints,  # ruff: ignore[private-member-access]
        "k": [Interval(Integral, 1, None, closed="left")],
        "estimator": [HasMethods(["fit", "predict_proba"]), None],
    }

    def __init__(
        self,
        k: int,
        estimator: BaseEstimator | None = None,
        *,
        transform_input=None,
        memory=None,
        verbose=False,
    ) -> None:
        """Assemble an entropy -> classifier pipeline pinned to `k` candidates."""
        self.assemble_steps(self.pooling, k, estimator, transform_input, memory, verbose)

    # A method rather than `__init__` code so that a class taking the pooling as a
    # constructor parameter can assemble the same pipeline: `__init__` signatures are the
    # `get_params` surface and cannot be shared.
    def assemble_steps(self, pooling, k, estimator, transform_input, memory, verbose) -> None:
        """Build the steps around `pooling`, a `Pooling` or a function."""
        if pooling is None:
            msg = (
                f"{type(self).__name__} fixes no pooling. Construct an EPRClassifier or a WEPRClassifier, "
                "or an EPR, a WEPR or a PoolingDetector."
            )
            raise TypeError(msg)
        Pipeline.__init__(
            self,
            steps=[("entropy", EntropyTransformer(pooling=pooling, k=k)), ("classifier", None)],
            transform_input=transform_input,
            memory=memory,
            verbose=verbose,
        )
        self.estimator = estimator

    @property
    def k(self) -> int:
        """Candidates per token, read from the first step.

        Written to every step before the classifier that takes a `k`, so that
        `set_params(k=)` reaches the steps that act on it instead of setting an attribute
        nothing consults. `set_params(entropy__k=)` changes one step only, and then that
        step alone reads fewer ranks than the others hand it, so sweep `k`, not the step
        parameter.

        Raises
        ------
        ValueError
            If the first step was replaced by one that has no `k`, such as `"passthrough"`.
        """
        name, step = self.steps[0]
        try:
            return step.k
        except AttributeError:
            msg = (
                f"The '{name}' step of {type(self).__name__} is {step!r}, which has no `k`: "
                "the first step must read a fixed number of candidates."
            )
            raise ValueError(msg) from None

    @k.setter
    def k(self, value: int) -> None:
        for _, step in self.steps[:-1]:
            if is_bearable(step, BaseEstimator) and "k" in step.get_params(deep=False):
                step.k = value

    @property
    def estimator(self) -> BaseEstimator:
        """The final estimator, which is the only fitted step in the pipeline.

        The constructor parameter of the same name, read back off the step it became, so
        that `get_params` and `clone` see the estimator actually in use rather than a copy
        of what was passed.
        """
        return self.steps[-1][1]

    @estimator.setter
    def estimator(self, value: BaseEstimator | None) -> None:
        # `None` is what `get_params` reports for the default classifier, so setting it back
        # (`set_params(**get_params())`, a grid over `[None, ...]`) restores the default
        # rather than leaving the final step empty. This is the one place it is built.
        self.steps[-1] = (self.steps[-1][0], default_classifier() if value is None else value)

    @classmethod
    def check_options(cls, kwargs: dict[str, Any]) -> None:
        """Refuse options that omit `k`, the width the weights were trained at.

        Raises
        ------
        TypeError
            If `"k"` is not a key of `kwargs`.
        """
        _require_k(kwargs)

    @classmethod
    def from_estimator(cls, estimator: BaseEstimator, identifier: str | Path, **kwargs: Any) -> "BaseEntropyClassifier":
        """An instance carrying `estimator` as its classifier, if the widths agree.

        Parameters
        ----------
        estimator : estimator
            The fitted estimator read from the published file.
        identifier : str or Path
            What named it, for the error below.
        **kwargs
            `k` (required), the candidates per token the estimator was trained at, and
            anything else `__init__` takes.

        Returns
        -------
        BaseEntropyClassifier
            An instance ready to `predict_proba`.

        Raises
        ------
        TypeError
            If `k` is missing from `kwargs`.
        ValueError
            If the estimator is not a fitted classifier with exactly two classes, or does
            not cover exactly `k` candidates. The check compares feature widths, and `EPR`
            has one feature at every `k`, so it cannot catch an `EPR` loaded with the wrong `k`.
        """
        try:
            check_is_fitted(estimator)
        except NotFittedError as error:
            msg = f"The estimator at '{identifier}' is not fitted, so it carries no weights to score with."
            raise ValueError(msg) from error
        if not is_classifier(estimator):
            msg = f"The estimator at '{identifier}' is a {type(estimator).__name__}, not a classifier."
            raise ValueError(msg)
        # Read through `Any`: a fitted classifier has `classes_`, which `BaseEstimator` does not declare.
        fitted: Any = estimator
        n_classes = len(fitted.classes_)
        if n_classes != 2:  # a detector scores P(class 1), which needs exactly two classes
            msg = (
                f"The estimator at '{identifier}' was fitted on {n_classes} class(es); a detector "
                "scores the probability of the second class of a binary classifier."
            )
            raise ValueError(msg)
        cls.check_options(kwargs)
        detector = cls(estimator=estimator, **kwargs)
        # The width the assembled pipeline hands its classifier, read from the pipeline's
        # own feature names so that the user's `k` and pooling decide it, never the file.
        expected = len(detector.named_steps["entropy"].get_feature_names_out())
        # `n_features_in_` is set by fit, so it is absent from the `BaseEstimator` interface
        # even though every estimator reaching here is fitted. Read it through `Any` rather
        # than suppressing per type-checker: the suppression is itself reported as unused by
        # versions that do not raise, which fails the hook the other way round.
        actual: int = fitted.n_features_in_
        if actual != expected:
            msg = (
                f"The {cls.__name__} detector at '{identifier}' takes {actual} feature(s), "
                f"but k={detector.k} gives {expected}. Its coefficients are fixed at the width "
                f"they were trained at; pass the k it was trained with."
            )
            raise ValueError(msg)
        return detector

    def __sklearn_tags__(self) -> Tags:
        """Declares a binary classifier over numeric 2-D tables, as the steps' tags cannot say."""
        tags = super().__sklearn_tags__()
        # `Pipeline` copies tags from its ends, which describe a 3-D cube input and a
        # multi-class-capable `LogisticRegression`; neither is this estimator's contract.
        # The entropy step is stateless as a transformer, but fitting it on a table is what
        # records the width and names that scikit-learn's checks verify.
        tags.no_validation = False
        tags.requires_fit = True
        tags.input_tags.allow_nan = False  # NaN pads a cube; a table is validated
        tags.classifier_tags.multi_class = False
        # A pooling whose width does not depend on `k` (EPR's one feature) cannot separate
        # classes that need more than a threshold, so its score on the checks' toy data is
        # allowed to be poor.
        tags.classifier_tags.poor_score = as_pooling(self.pooling).constant_width
        tags.input_tags.two_d_array = True
        tags.input_tags.three_d_array = False
        return tags

    @_fit_context(prefer_skip_nested_validation=False)  # the steps are validated by their own fit
    def fit(self, X, y=None, **params) -> "BaseEntropyClassifier":  # ruff: ignore[invalid-argument-name]
        """Fit the entropy and classifier steps; targets must be binary.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_tokens, width) or (n_samples, width)
            Per-rank logprobs, NaN-padded in the 3-D form; a 2-D table is one token per row.
        y : array-like of shape (n_samples,), default=None
            Binary labels. Any two distinct labels are accepted, and the second in sorted
            order, `classes_[1]`, is the class `predict_proba(...)[:, 1]` scores; use
            `0`/`1` with 1 marking the hallucination class.
        **params
            Passed to `Pipeline.fit`.

        Returns
        -------
        self : BaseEntropyClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            If a constructor parameter is invalid; if `y` is not 1-D-compatible, holds NaN or
            infinity, is continuous, or does not have exactly two classes; or if the entropy
            step's `pooling` is not the one this class fixes.
        """
        entropy = self.named_steps.get("entropy")
        if is_bearable(entropy, EntropyTransformer) and entropy.pooling != self.pooling:
            msg = (
                f"{type(self).__name__} scores with pooling={self.pooling!r}, but its entropy step "
                f"has pooling={entropy.pooling!r}, which would fit coefficients the class "
                "cannot describe. Use the class built for that pooling, or a subclass that sets `pooling`; "
                "for responses, a PoolingDetector takes any pooling."
            )
            raise ValueError(msg)
        if y is not None:
            y = column_or_1d(y, warn=True)
            # `check_classification_targets` accepts NaN in float labels as "unknown".
            if y.dtype.kind in "fc" and not np.all(np.isfinite(y)):
                msg = "Input y contains NaN or infinity."
                raise ValueError(msg)
            check_classification_targets(y)
            kind = type_of_target(y)
            if kind != "binary":
                msg = f"Only binary classification is supported. The type of the target is {kind}."
                raise ValueError(msg)
        # `Pipeline._fit` rebinds `self.steps` to a copy, which `check_dont_overwrite_parameters`
        # reads as a mutated constructor argument. Restoring the original list keeps the
        # object `clone` and `get_params` hand around the same one.
        steps = self.steps
        try:
            return super().fit(X, y, **params)
        finally:
            steps[:] = self.steps
            self.steps = steps

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # ruff: ignore[boolean-type-hint-positional-argument, boolean-default-value-positional-argument]
        """`Pipeline.get_params`, reporting `estimator=None` for the untouched default.

        Parameters
        ----------
        deep : bool, default=True
            Whether to include the parameters of the steps, as `Pipeline.get_params`.

        Returns
        -------
        dict
            Parameter names mapped to their values.

        Notes
        -----
        `estimator` is a property reading the final step, so it would always report the
        default classifier instance the constructor built, and `clone` would then differ
        from the constructor argument `None` (sklearn's `clone` checks identity). While
        the final step is still a default `LogisticRegression`, `None` is reported, and the
        rebuilt estimator builds the same default. A step that was replaced or had a
        parameter changed through `set_params` is reported as itself. With `estimator`
        reported as `None`, its `estimator__*` sub-parameters are omitted, as their
        `classifier__*` twins already carry them.
        """
        params = super().get_params(deep=deep)
        step = self.steps[-1][1]
        if type(step) is LogisticRegression and step.get_params() == default_classifier().get_params():
            params["estimator"] = None
            # A `None` has no parameters to set; its default's are still reachable as
            # `classifier__*`, and `set_params` would otherwise route `estimator__*` to it.
            params = {name: value for name, value in params.items() if not name.startswith("estimator__")}
        return params

    def set_params(self, **params) -> "BaseEntropyClassifier":
        """`Pipeline.set_params`, where `estimator__<p>` sets `<p>` on the final step.

        `get_params` reports `estimator=None` while the final step is the untouched default,
        and the base class would route `estimator__<p>` to that `None`. The final step is the
        estimator, so the key is addressed to it directly. A call that also replaces
        `estimator` keeps its keys as given, so they reach the replacement.
        """
        if "estimator" not in params:
            final = self.steps[-1][0]
            params = {
                (f"{final}__{name[len('estimator__') :]}" if name.startswith("estimator__") else name): value
                for name, value in params.items()
            }
        return super().set_params(**params)

    def __getitem__(self, ind):
        """A step, or a plain `Pipeline` over a slice of them.

        `Pipeline.__getitem__` rebuilds `self.__class__` from a list of steps, which this
        constructor does not take: the estimator is the whole chain, and any slice of it is
        something else. The slice is delegated to a `Pipeline` over the same steps, which is
        what makes `estimator[:-1].transform(...)` -- the features without the classifier --
        work, and applies `Pipeline`'s own slice rules (a step of 1 only).
        """
        if is_bearable(ind, slice):
            whole = Pipeline(self.steps, memory=self.memory, verbose=self.verbose, transform_input=self.transform_input)
            return whole[ind]
        return super().__getitem__(ind)

    def predict_token_proba(self, X) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Per-token hallucination probabilities, for locating *where* a response drifts.

        Runs the transformer steps in token mode (`transform_tokens`, falling back to
        `transform`), then scores only the non-padded token rows and scatters the results
        back, so padded positions stay NaN rather than being scored as real tokens.

        Parameters
        ----------
        X : array-like
            The same input `predict_proba` accepts.

        Returns
        -------
        ndarray of shape (n_sequences, max_tokens, 1)
            Probability of class 1 per token, NaN at padded positions; all NaN, and
            `(0, 0, 1)` for an empty batch, when no token is real.

        Raises
        ------
        ValueError
            If a step reduces the token axis away, so token mode yields fewer than three
            dimensions.
        """
        raw_output = X
        sequence_mode = []  # named in the error below, from the same probe that picks the method
        for name, transformer in self.steps[:-1]:
            try:
                step_transform = transformer.transform_tokens
            except AttributeError:
                step_transform = transformer.transform
                sequence_mode.append(name)
            raw_output = step_transform(raw_output)

        token_features = np.asarray(raw_output)

        # A step that reduces over the token axis in `transform` and declares no
        # `transform_tokens` is driven through `transform` here, and silently hands back
        # sequence-level features. Unpacking that below fails on the arity alone, naming
        # neither the step nor the reason, so the shape is read first.
        if token_features.ndim != 3:
            reduced = ", ".join(sequence_mode)
            msg = (
                f"Token mode produced a {token_features.ndim}D array, expected 3D "
                f"(n_sequences, max_tokens, n_features). A pipeline step reduced the token axis away: "
                f"{reduced or 'no step'} declares no transform_tokens, so it ran in sequence mode."
            )
            raise ValueError(msg)

        n_samples, max_tokens, n_features = token_features.shape
        flat_features = np.asarray(token_features).reshape(n_samples * max_tokens, n_features)
        non_padded = ~np.isnan(flat_features).any(axis=1)

        classifier = self.steps[-1][1]
        flat_scores = np.full(n_samples * max_tokens, np.nan)
        if non_padded.any():  # the classifier refuses zero rows, which an empty or fully padded batch gives
            flat_scores[non_padded] = classifier.predict_proba(flat_features[non_padded])[:, 1]

        return flat_scores.reshape(n_samples, max_tokens, 1)


class EPRClassifier(BaseEntropyClassifier):
    """Scores logprob arrays with one pooled feature, the Entropy Production Rate.

    EPR pools every candidate of the token distribution into one number, so the
    calibration fits one coefficient, whatever `k` is. Input is a `(n, tokens, width)`
    logprob array, or a numeric 2-D table read as one token per row. Targets are binary,
    1 marking hallucination.

    Both poolings need weights fit on labelled data, so choosing this one saves no setup
    work over `WEPRClassifier` -- only parameters. Prefer `WEPRClassifier` unless there is
    too little labelled data to fit its larger coefficient vector. To score completion
    responses, use `EPR`.

    Parameters
    ----------
    k : int
        Candidates per token read. See `BaseEntropyClassifier`.
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
    classes_ : ndarray of shape (2,)
        The class labels, sorted. Available once fitted.
    n_features_in_ : int
        Width of the table, or last dimension of the 3-D array, fitted on.
    named_steps : Bunch
        The steps by name: `"entropy"` and `"classifier"`.
    steps : list of (str, estimator)
        The same steps, in order.
    estimator : estimator
        The final step.

    See Also
    --------
    WEPRClassifier : Reads each rank separately, at the cost of `2k` coefficients.
    EPR : The same pooling over completion responses.

    Examples
    --------
    >>> classifier = EPRClassifier(k=15).fit(logprobs, y)  # doctest: +SKIP
    >>> classifier.predict_proba(logprobs)[:, 1]  # doctest: +SKIP
    """

    pooling: ClassVar[Pooling | None] = EPRPooling()


class WEPRClassifier(BaseEntropyClassifier):
    """Scores logprob arrays with a mean and a max per rank, the Weighted EPR.

    WEPR has `2k` features, letting the calibration weight the informative ranks over the
    rest -- the second-most-likely candidate carries different information from the
    tenth, and a coefficient per rank is what lets the fit say so. It reads strictly more
    of the distribution than EPR, which makes it the more accurate of the two when there
    is enough labelled data to fit `2k` coefficients. Input is a `(n, tokens, width)`
    logprob array, or a numeric 2-D table read as one token per row. Targets are binary,
    1 marking hallucination. To score completion responses, use `WEPR`.

    Parameters
    ----------
    k : int
        Candidates per token read. See `BaseEntropyClassifier`.
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
    classes_ : ndarray of shape (2,)
        The class labels, sorted. Available once fitted.
    n_features_in_ : int
        Width of the table, or last dimension of the 3-D array, fitted on.
    named_steps : Bunch
        The steps by name: `"entropy"` and `"classifier"`.
    steps : list of (str, estimator)
        The same steps, in order.
    estimator : estimator
        The final step.

    See Also
    --------
    EPRClassifier : One pooled feature, for scarce labelled data.
    WEPR : The same pooling over completion responses.

    Examples
    --------
    >>> classifier = WEPRClassifier(k=15).fit(logprobs, y)  # doctest: +SKIP
    >>> classifier.predict_proba(logprobs)[:, 1]  # doctest: +SKIP
    """

    pooling: ClassVar[Pooling | None] = WEPRPooling()
