"""The shipped detectors over logprob arrays: `EPR` and `WEPR`."""

from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from beartype.door import is_bearable
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context, clone
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import Tags
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import assert_all_finite, check_is_fitted, column_or_1d

from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.scoring.entropy_methods.pooling import EPRPooling, WEPRPooling
from artefactual.scoring.tokens import TokenProbaMixin
from artefactual.utils.io import PretrainedMixin, SavePretrainedMixin, _require_k, check_loaded_classifier
from artefactual.utils.typing import TYPE_CHECKED, LogprobArray


def default_classifier() -> LogisticRegression:
    """The unregularised logistic regression the published detectors were fit with.

    Regularisation is switched off with `C=np.inf`.
    """
    return LogisticRegression(C=np.inf, max_iter=1000)


def check_binary_target(y) -> None:
    """Refuse targets that are not binary labels.

    Raises
    ------
    ValueError
        If `y` is not 1-D-compatible, holds NaN or infinity, is continuous, or does not have
        exactly two classes.
    """
    y = column_or_1d(y, warn=True)
    assert_all_finite(y, input_name="y")
    check_classification_targets(y)
    kind = type_of_target(y)
    if kind != "binary":
        msg = f"Only binary classification is supported. The type of the target is {kind}."
        raise ValueError(msg)


def check_width(X, k: int) -> Any:  # ruff: ignore[invalid-argument-name]
    """Return `X` unchanged after refusing input whose last dimension is not `k`.

    Raises
    ------
    ValueError
        If `X` has at least two dimensions, a first one that is not empty, and a last one that is not `k`.
    """
    shape = np.asarray(X).shape  # not np.shape: it refuses array-likes that only define `__array__`
    if len(shape) >= 2 and shape[0] and shape[-1] != k:  # an empty batch has no width: the parser emits (0, 0, 0)
        msg = f"trained on {k} candidates per token, got {shape[-1]}: use LogProbParser(k={k})"
        raise ValueError(msg)
    return X


def binary_tags(tags: Tags, *, poor_score: bool) -> Tags:
    """`tags` of a binary classifier over numeric tables and NaN-padded cubes."""
    tags.input_tags.allow_nan = False  # a table is validated; only a cube takes NaN as padding
    tags.input_tags.two_d_array = True
    tags.input_tags.three_d_array = False
    if tags.classifier_tags is not None:
        tags.classifier_tags.multi_class = False
        # A pooling of constant width cannot separate classes that need more than a threshold.
        tags.classifier_tags.poor_score = poor_score
    return tags


class EPR(TokenProbaMixin, PretrainedMixin, SavePretrainedMixin, ClassifierMixin, BaseEstimator):
    """Scores logprob arrays with one pooled feature, the Entropy Production Rate.

    Class 1 is the hallucination class: `predict_proba(...)[:, 1]` is the score of
    interest. Input is the `(n, tokens, width)` array `LogProbParser` emits, or a numeric
    2-D `(n, width)` table read as one token per row. Targets are binary. To score
    completion responses, put a `LogProbParser` in front.

    Parameters
    ----------
    estimator : estimator, default=None
        Final classifier with `fit` and `predict_proba`, never fitted in place: a clone is
        trained. `None` is the unregularised logistic regression the published detectors
        were fit with. A `FrozenEstimator` holds pretrained weights, which `fit` keeps.
    k : int, default=None
        Candidates per token the detector accepts. `None` accepts any width. EPR pools to one
        feature whatever the width, so `k` is the only guard against input from another `k`.

    Attributes
    ----------
    fitted_pipeline_ : Pipeline
        The pipeline `fit` trained; absent on a detector that only carries frozen weights.
    pipeline_ : Pipeline
        The pipeline that scores: the width check when `k` is set, the entropy step and the
        classifier. `fitted_pipeline_` once `fit` ran, else rebuilt from a `FrozenEstimator`.
    classes_ : ndarray of shape (2,)
        The class labels, sorted; column `i` of `predict_proba` is class `classes_[i]`.
    n_features_in_ : int
        Last dimension of the input fitted on.

    Raises
    ------
    TypeError
        From `fit` and `predict*`, if given completion responses instead of logprob arrays.
    ValueError
        From `fit` and `predict*`, if `k` is set and the input has another width.

    Examples
    --------
    >>> detector = make_pipeline(LogProbParser(k=15), EPR(k=15)).fit(
    ...     responses, y
    ... )  # doctest: +SKIP
    >>> detector = make_pipeline(
    ...     LogProbParser(k=15), EPR.from_pretrained("artefactory/epr-phi4", k=15)
    ... )  # doctest: +SKIP
    """

    _parameter_constraints = {  # ruff: ignore[mutable-class-default]
        "estimator": [HasMethods(["fit", "predict_proba"]), None],
        "k": [Interval(Integral, 1, None, closed="left"), None],
    }

    def __init__(self, estimator: BaseEstimator | None = None, k: int | None = None) -> None:
        self.estimator = estimator
        self.k = k

    @classmethod
    def check_options(cls, kwargs) -> None:
        """Require `k`, the width the loaded weights were trained at."""
        _require_k(kwargs)

    @classmethod
    def from_estimator(cls, estimator: BaseEstimator, identifier: str | Path, **kwargs: Any) -> "EPR":
        """Wrap a loaded classifier, frozen, as a detector.

        Parameters
        ----------
        estimator : estimator
            A fitted binary classifier.
        identifier : str or Path
            What named it, for error messages.
        **kwargs
            Passed to the constructor.

        Returns
        -------
        EPR
            A detector whose `estimator` is `FrozenEstimator(estimator)`.

        Raises
        ------
        ValueError
            If `estimator` is not a fitted binary classifier.
        """
        check_loaded_classifier(estimator, identifier)
        return cls(estimator=FrozenEstimator(estimator), **kwargs)

    def __sklearn_tags__(self) -> Tags:
        """A binary classifier over numeric tables; one pooled feature separates little."""
        return binary_tags(super().__sklearn_tags__(), poor_score=True)

    # Private: `fit` and `pipeline_` must build the same steps, or a loaded detector would skip the check.
    def _build_pipeline(self, classifier: BaseEstimator) -> Pipeline:
        """The width check when `k` is set, the entropy step, then `classifier`."""
        steps = [EntropyTransformer(EPRPooling()), classifier]
        if self.k is not None:
            steps.insert(0, FunctionTransformer(check_width, kw_args={"k": self.k}, feature_names_out="one-to-one"))
        return make_pipeline(*steps)

    @_fit_context(prefer_skip_nested_validation=False)  # the pipeline validates its own steps
    @beartype(conf=TYPE_CHECKED)
    def fit(self, X: LogprobArray, y) -> "EPR":  # ruff: ignore[invalid-argument-name]
        """Fit a clone of `estimator` on the pooled features; a frozen one keeps its weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_tokens, width) or (n_samples, width)
            Per-rank logprobs, NaN-padded in the 3-D form.
        y : array-like of shape (n_samples,)
            Binary labels; the second in sorted order is class 1.

        Returns
        -------
        self : EPR
        """
        check_binary_target(y)
        self.fitted_pipeline_ = self._build_pipeline(
            clone(default_classifier() if self.estimator is None else self.estimator)
        ).fit(X, y)
        entropy_step = self.fitted_pipeline_[-2]
        self.n_features_in_ = entropy_step.n_features_in_
        if "feature_names_in_" in vars(entropy_step):
            self.feature_names_in_ = entropy_step.feature_names_in_
        return self

    @property
    def pipeline_(self) -> Pipeline:
        """The entropy step and the classifier that score.

        The pipeline `fit` trained once it ran. A `FrozenEstimator` was never fitted here, so
        the pipeline is rebuilt from the parameters on each read, never stored.

        Raises
        ------
        AttributeError
            If neither holds: the detector is not fitted.
        """
        if "fitted_pipeline_" in vars(self):
            return self.fitted_pipeline_
        if is_bearable(self.estimator, FrozenEstimator):
            return self._build_pipeline(self.estimator)
        msg = f"'{type(self).__name__}' object has no attribute 'pipeline_': it is not fitted"
        raise AttributeError(msg)

    @property
    def classes_(self) -> np.ndarray:
        """The class labels, sorted; column `i` of `predict_proba` is `classes_[i]`.

        Raises
        ------
        AttributeError
            If the detector is not fitted.
        """
        return self.pipeline_.classes_

    @beartype(conf=TYPE_CHECKED)
    def predict(self, X: LogprobArray) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Predicted labels."""
        check_is_fitted(self)
        return self.pipeline_.predict(X)

    @beartype(conf=TYPE_CHECKED)
    def predict_proba(self, X: LogprobArray) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Class probabilities; column `i` is `classes_[i]`."""
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)

    @beartype(conf=TYPE_CHECKED)
    def decision_function(self, X: LogprobArray) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """The classifier's decision function."""
        check_is_fitted(self)
        return self.pipeline_.decision_function(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the pooled features."""
        check_is_fitted(self)
        return self.pipeline_[-2].get_feature_names_out(input_features)


class WEPR(TokenProbaMixin, PretrainedMixin, SavePretrainedMixin, ClassifierMixin, BaseEstimator):
    """Scores logprob arrays with a mean and a max per rank, the Weighted EPR.

    Two features per rank let the calibration weight the informative ranks over the rest,
    so it reads strictly more of the distribution than `EPR`. Input and targets are as for
    `EPR`; the weights are fixed at the width they were trained at, which the pooling's
    feature count then enforces.

    Parameters
    ----------
    estimator : estimator, default=None
        Final classifier with `fit` and `predict_proba`, never fitted in place: a clone is
        trained. `None` is the unregularised logistic regression. A `FrozenEstimator` holds
        pretrained weights, which `fit` keeps.

    Attributes
    ----------
    fitted_pipeline_ : Pipeline
        The pipeline `fit` trained; absent on a detector that only carries frozen weights.
    pipeline_ : Pipeline
        The pipeline that scores: the entropy step and the classifier.
        `fitted_pipeline_` once `fit` ran, else rebuilt from a `FrozenEstimator`.
    classes_ : ndarray of shape (2,)
        The class labels, sorted.
    n_features_in_ : int
        Last dimension of the input fitted on.

    Raises
    ------
    TypeError
        From `fit` and `predict*`, if given completion responses instead of logprob arrays.

    Examples
    --------
    >>> detector = make_pipeline(LogProbParser(k=15), WEPR()).fit(
    ...     responses, y
    ... )  # doctest: +SKIP
    """

    _parameter_constraints = {  # ruff: ignore[mutable-class-default]
        "estimator": [HasMethods(["fit", "predict_proba"]), None],
    }

    def __init__(self, estimator: BaseEstimator | None = None) -> None:
        self.estimator = estimator

    @classmethod
    def from_estimator(cls, estimator: BaseEstimator, identifier: str | Path, **kwargs: Any) -> "WEPR":
        """Wrap a loaded classifier, frozen, as a detector.

        Parameters
        ----------
        estimator : estimator
            A fitted binary classifier.
        identifier : str or Path
            What named it, for error messages.
        **kwargs
            Passed to the constructor.

        Returns
        -------
        WEPR
            A detector whose `estimator` is `FrozenEstimator(estimator)`.

        Raises
        ------
        ValueError
            If `estimator` is not a fitted binary classifier.
        """
        check_loaded_classifier(estimator, identifier)
        return cls(estimator=FrozenEstimator(estimator), **kwargs)

    def __sklearn_tags__(self) -> Tags:
        """A binary classifier over numeric tables."""
        return binary_tags(super().__sklearn_tags__(), poor_score=False)

    def _build_pipeline(self, classifier: BaseEstimator) -> Pipeline:
        """The entropy step, then `classifier`."""
        return make_pipeline(EntropyTransformer(WEPRPooling()), classifier)

    @_fit_context(prefer_skip_nested_validation=False)  # the pipeline validates its own steps
    @beartype(conf=TYPE_CHECKED)
    def fit(self, X: LogprobArray, y) -> "WEPR":  # ruff: ignore[invalid-argument-name]
        """Fit a clone of `estimator` on the pooled features; a frozen one keeps its weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_tokens, width) or (n_samples, width)
            Per-rank logprobs, NaN-padded in the 3-D form.
        y : array-like of shape (n_samples,)
            Binary labels; the second in sorted order is class 1.

        Returns
        -------
        self : WEPR
        """
        check_binary_target(y)
        self.fitted_pipeline_ = self._build_pipeline(
            clone(default_classifier() if self.estimator is None else self.estimator)
        ).fit(X, y)
        entropy_step = self.fitted_pipeline_[-2]
        self.n_features_in_ = entropy_step.n_features_in_
        if "feature_names_in_" in vars(entropy_step):
            self.feature_names_in_ = entropy_step.feature_names_in_
        return self

    @property
    def pipeline_(self) -> Pipeline:
        """The entropy step and the classifier that score.

        The pipeline `fit` trained once it ran. A `FrozenEstimator` was never fitted here, so
        the pipeline is rebuilt from the parameters on each read, never stored.

        Raises
        ------
        AttributeError
            If neither holds: the detector is not fitted.
        """
        if "fitted_pipeline_" in vars(self):
            return self.fitted_pipeline_
        if is_bearable(self.estimator, FrozenEstimator):
            return self._build_pipeline(self.estimator)
        msg = f"'{type(self).__name__}' object has no attribute 'pipeline_': it is not fitted"
        raise AttributeError(msg)

    @property
    def classes_(self) -> np.ndarray:
        """The class labels, sorted; column `i` of `predict_proba` is `classes_[i]`.

        Raises
        ------
        AttributeError
            If the detector is not fitted.
        """
        return self.pipeline_.classes_

    @beartype(conf=TYPE_CHECKED)
    def predict(self, X: LogprobArray) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Predicted labels."""
        check_is_fitted(self)
        return self.pipeline_.predict(X)

    @beartype(conf=TYPE_CHECKED)
    def predict_proba(self, X: LogprobArray) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """Class probabilities; column `i` is `classes_[i]`."""
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)

    @beartype(conf=TYPE_CHECKED)
    def decision_function(self, X: LogprobArray) -> np.ndarray:  # ruff: ignore[invalid-argument-name]
        """The classifier's decision function."""
        check_is_fitted(self)
        return self.pipeline_.decision_function(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the pooled features."""
        check_is_fitted(self)
        return self.pipeline_[-2].get_feature_names_out(input_features)
