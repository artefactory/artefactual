from sklearn.base import BaseEstimator, TransformerMixin

from artefactual import parse_top_logprobs


class LogProbParser(BaseEstimator, TransformerMixin):
    """
    Wraps parse_top_logprobs as a pipeline step.

    Because this transformer accepts non-standard list[dict] inputs
    by bypassing scikit-learn validation, cross-validation
    must start from data that has already been parsed.
    """

    def __init__(self):
        pass

    def fit(self, _x, _y=None) -> "LogProbParser":
        return self

    def transform(self, x: list) -> list[dict[int, list[float]]]:
        return parse_top_logprobs(x)

    def _more_tags(self) -> dict[str, bool]:
        """
        Bypasses strict scikit-learn array validation checks, allowing
        the transformer to accept a raw list of dictionaries.
        """
        return {"no_validation": True}
