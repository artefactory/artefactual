"""The `EPR` / `WEPR` detectors, which score completion responses, and the deprecated factories."""

from typing import ClassVar

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils import Tags

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_classifier import DEFAULT_K, BaseEntropyClassifier
from artefactual.utils.io import Reduction


class BaseDetector(BaseEntropyClassifier):
    """A `parser -> entropy -> classifier` pipeline returning P(hallucination) from responses.

    A `BaseEntropyClassifier` with a `LogProbParser` in front, so it takes completion
    responses rather than logprob arrays. It keeps everything the classifier does -- the
    parameters, targets, persistence and `predict_token_proba` -- and adds only the parsing.

    Abstract in the reduction: `EPR` and `WEPR` are the detectors to construct.

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
        If constructed directly rather than through `EPR` or `WEPR`.

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
        self.steps.insert(0, ("parser", LogProbParser(k=k)))

    def __sklearn_tags__(self) -> Tags:
        """Declares a binary classifier over responses, which are not an array."""
        # The classifier's tags describe a numeric table, and the front of this pipeline
        # takes none; the parser's own tags are what `Pipeline` reports for its input.
        tags = Pipeline.__sklearn_tags__(self)
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = self.reduction == "epr"
        return tags

    @classmethod
    def trainable(
        cls, reduction: Reduction, k: int = DEFAULT_K, *, estimator: BaseEstimator | None = None
    ) -> "BaseDetector":
        """An unfitted detector, selecting the reduction by name.

        For callers that hold the reduction as data -- a CLI argument, a column in a sweep
        -- so that selecting one never means re-deriving the mapping from its name to a
        class.

        Parameters
        ----------
        reduction : {"epr", "wepr"}
            The reduction, which selects the class.
        k : int, default=15
            Candidates per token the responses carry.
        estimator : estimator, default=None
            Final estimator to fit. Defaults to the unregularised logistic regression the
            published detectors were fit with.

        Returns
        -------
        BaseDetector
            A detector ready to `fit`.
        """
        return {"epr": EPR, "wepr": WEPR}[reduction](k=k, estimator=estimator)


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
    >>> detector = EPR().fit(responses, y)  # doctest: +SKIP
    >>> published = EPR.from_pretrained("artefactory/epr-phi4")  # doctest: +SKIP
    >>> published.predict_proba(response)[:, 1]  # doctest: +SKIP
    """

    reduction: ClassVar[Reduction | None] = "epr"


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
    >>> detector = WEPR().fit(responses, y)  # doctest: +SKIP
    >>> published = WEPR.from_pretrained("artefactory/wepr-phi4")  # doctest: +SKIP
    >>> published.predict_proba(response)[:, 1]  # doctest: +SKIP
    """

    reduction: ClassVar[Reduction | None] = "wepr"
