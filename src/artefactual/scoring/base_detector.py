"""The detector pipeline and the `EPR` / `WEPR` detectors built on it."""

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_methods.entropy_transformer import EntropyTransformer
from artefactual.utils.io import EstimatorPersistenceMixin, Reduction

# Every published detector was fit at 15 ranks.
DEFAULT_K = 15


class BaseDetector(Pipeline, EstimatorPersistenceMixin):
    """A `parser -> entropy -> classifier` pipeline returning P(hallucination).

    A scikit-learn `Pipeline`, so `predict`, `predict_proba`, `fit`, `get_params` and
    `clone` behave as expected and the detector composes into `GridSearchCV` and friends.

    Abstract in the reduction: `EPR` and `WEPR` are the detectors to construct. Each
    subclass fixes `reduction` and the coefficient width that reduction implies, which is
    all that distinguishes one detector from another.

    Class 1 is the hallucination class: `predict_proba(...)[:, 1]` is the score of
    interest.
    """

    #: The entropy reduction this detector scores with. Set by each subclass.
    reduction: ClassVar[Reduction | None] = None

    def __init__(
        self,
        k: int = DEFAULT_K,
        estimator: BaseEstimator | None = None,
        *,
        transform_input=None,
        memory=None,
        verbose=False,
    ) -> None:
        """Assemble a parser -> entropy -> classifier pipeline pinned to `k` ranks.

        `k` is handled at the ends of the pipeline: the parser sizes the rank axis to it,
        and any loaded weights were checked against it beforehand. The entropy step in
        between carries no rank count, since its input width is already `k`.

        Args:
            k: Rank count the responses carry. Responses carrying fewer are rejected when
                parsed, rather than padded, since the missing ranks were never fetched.
            estimator: Final estimator. Unfitted by default -- the unregularised logistic
                regression the published detectors were fit with, so coefficients fitted
                here are comparable to the shipped ones. `C=np.inf` rather than
                `penalty=None`: the latter is deprecated in scikit-learn 1.8 and removed in
                1.10, and the two produce identical coefficients.
            transform_input: Passed to `Pipeline`.
            memory: Passed to `Pipeline`.
            verbose: Passed to `Pipeline`.

        Raises:
            TypeError: If constructed directly rather than through `EPR` or `WEPR`.
        """
        if self.reduction is None:
            msg = f"{type(self).__name__} fixes no reduction. Construct an EPR or a WEPR detector."
            raise TypeError(msg)
        super().__init__(
            steps=[
                ("parser", LogProbParser(k=k)),
                ("entropy", EntropyTransformer(reduction=self.reduction)),
                ("classifier", estimator if estimator is not None else LogisticRegression(C=np.inf, max_iter=1000)),
            ],
            transform_input=transform_input,
            memory=memory,
            verbose=verbose,
        )

    @property
    def k(self) -> int:
        """Rank count the parser reads, and the width the reduction covers.

        Read from the parser step rather than stored alongside it, so that `set_params(k=)`
        -- which is how `GridSearchCV` sweeps it -- reaches the step that acts on it
        instead of setting an attribute nothing consults.
        """
        return self.named_steps["parser"].k

    @k.setter
    def k(self, value: int) -> None:
        self.named_steps["parser"].k = value

    @classmethod
    def _feature_count(cls, k: int) -> int:
        """Features this reduction produces at `k` ranks.

        What a loaded estimator's coefficient vector is checked against: a detector's
        coefficients are fixed at the rank count they were trained at.
        """
        raise NotImplementedError

    @classmethod
    def _from_estimator(cls, estimator: BaseEstimator, identifier: str | Path, **kwargs: Any) -> "BaseDetector":
        """A detector carrying `estimator` as its classifier, if the widths agree.

        Args:
            estimator: The fitted estimator read from the published file.
            identifier: What named it, for the error below.
            **kwargs: `k`, and anything else `__init__` takes.

        Returns:
            A detector ready to `predict_proba`.

        Raises:
            ValueError: If the estimator does not cover exactly `k` ranks.
        """
        k = kwargs.get("k", DEFAULT_K)
        expected = cls._feature_count(k)
        # `n_features_in_` is set by fit, so it is absent from the `BaseEstimator` interface
        # even though every estimator reaching here is fitted. Read it through `Any` rather
        # than suppressing per type-checker: the suppression is itself reported as unused by
        # versions that do not raise, which fails the hook the other way round.
        fitted: Any = estimator
        actual: int = fitted.n_features_in_
        if actual != expected:
            msg = (
                f"The {cls.__name__} detector at '{identifier}' takes {actual} feature(s), "
                f"but k={k} needs {expected}. Its coefficients are fixed at the rank count "
                f"they were trained at; pass k={cls._implied_k(actual)}, or use a detector "
                f"trained at k={k}."
            )
            raise ValueError(msg)
        return cls(estimator=estimator, **kwargs)

    @classmethod
    def _implied_k(cls, n_features: int) -> int:
        """The rank count `n_features` coefficients were trained at."""
        raise NotImplementedError

    @property
    def estimator(self) -> BaseEstimator:
        """The final estimator, which is the only fitted step in the pipeline.

        The constructor parameter of the same name, read back off the step it became, so
        that `get_params` and `clone` see the estimator actually in use rather than a copy
        of what was passed.
        """
        return self.steps[-1][1]

    @estimator.setter
    def estimator(self, value: BaseEstimator) -> None:
        self.steps[-1] = (self.steps[-1][0], value)

    def __getitem__(self, ind):
        """A step, or a plain `Pipeline` over a slice of them.

        `Pipeline.__getitem__` rebuilds `self.__class__` from a list of steps, which a
        detector's constructor does not take: a detector is the whole
        parser -> entropy -> classifier chain, and any slice of it is something else. The
        slice is returned as the `Pipeline` it is, which is what makes
        `detector[:-1].transform(...)` -- the features without the classifier -- work.
        """
        if isinstance(ind, slice):
            return Pipeline(self.steps[ind], memory=self.memory, verbose=self.verbose)
        return super().__getitem__(ind)

    @classmethod
    def trainable(
        cls, reduction: Reduction, k: int = DEFAULT_K, *, estimator: BaseEstimator | None = None
    ) -> "BaseDetector":
        """An unfitted detector, selecting the reduction by name.

        For callers that hold the reduction as data -- a CLI argument, a column in a sweep
        -- so that selecting one never means re-deriving the mapping from its name to a
        class.

        Args:
            reduction: `"epr"` or `"wepr"`.
            k: Rank count the responses carry.
            estimator: Final estimator to fit. Defaults to the unregularised logistic
                regression the published detectors were fit with.

        Returns:
            A detector ready to `fit`.
        """
        return {"epr": EPR, "wepr": WEPR}[reduction](k=k, estimator=estimator)

    def predict_token_proba(self, x) -> np.ndarray:
        """Per-token hallucination probabilities, for locating *where* a response drifts.

        Runs the transformer steps in token mode (`transform_tokens`, falling back to
        `transform`), then scores only the non-padded token rows and scatters the results
        back, so padded positions stay NaN rather than being scored as real tokens.

        Args:
            x: The same input `predict_proba` accepts.

        Returns:
            `(n_sequences, max_tokens, 1)`, NaN at padded positions.
        """
        raw_output = x
        for _, transformer in self.steps[:-1]:
            try:
                step_transform = transformer.transform_tokens
            except AttributeError:
                step_transform = transformer.transform
            raw_output = step_transform(raw_output)

        token_features = np.asarray(raw_output)

        # A step that reduces over the token axis in `transform` and declares no
        # `transform_tokens` is driven through `transform` here, and silently hands back
        # sequence-level features. Unpacking that below fails on the arity alone, naming
        # neither the step nor the reason, so the shape is read first.
        if token_features.ndim != 3:
            reduced = ", ".join(name for name, step in self.steps[:-1] if not hasattr(step, "transform_tokens"))
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
        flat_scores[non_padded] = classifier.predict_proba(flat_features[non_padded])[:, 1]

        return flat_scores.reshape(n_samples, max_tokens, 1)


class EPR(BaseDetector):
    """A detector that pools a response's uncertainty into one number.

    EPR -- Entropy Production Rate. A single feature, pooling every rank of the token
    distribution into one number, so the calibration fits one coefficient.

    Both detectors need weights fit on labelled data, so choosing this one saves no setup
    work over `WEPR` -- only parameters. Prefer `WEPR` unless there is too little labelled
    data to fit its larger coefficient vector.

    Example:
        >>> detector = EPR().fit(responses, y)  # doctest: +SKIP
        >>> published = EPR.from_pretrained("artefactory/epr-phi4")  # doctest: +SKIP
        >>> published.predict_proba(response)[:, 1]  # doctest: +SKIP
    """

    reduction: ClassVar[Reduction | None] = "epr"

    @classmethod
    def _feature_count(cls, k: int) -> int:  # noqa: ARG003 — EPR pools every rank into one feature, whatever k is
        return 1

    @classmethod
    def _implied_k(cls, n_features: int) -> int:
        return n_features


class WEPR(BaseDetector):
    """A detector that reads each rank of the token distribution.

    WEPR -- Weighted EPR. `2k` features, a mean and a max per rank, letting the
    calibration weight the informative ranks over the rest. It reads strictly more of the
    distribution than `EPR` at the same calibration cost, which makes it the default
    choice.

    Example:
        >>> detector = WEPR().fit(responses, y)  # doctest: +SKIP
        >>> published = WEPR.from_pretrained("artefactory/wepr-phi4")  # doctest: +SKIP
        >>> published.predict_proba(response)[:, 1]  # doctest: +SKIP
    """

    reduction: ClassVar[Reduction | None] = "wepr"

    @classmethod
    def _feature_count(cls, k: int) -> int:
        return 2 * k

    @classmethod
    def _implied_k(cls, n_features: int) -> int:
        return n_features // 2
