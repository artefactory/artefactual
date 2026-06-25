import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import Tags


class PretrainedLogisticRegression(LogisticRegression):
    def __sklearn_tags__(self) -> Tags:
        """
        Fetch the default LogisticRegression tags and override validation
        to allow passing raw lists of dicts or custom inputs.
        """
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        return tags

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Path | str) -> "PretrainedLogisticRegression":
        """
        Instantiate and initialize a LogisticRegression model using pre-trained weights.
        The metric (EPR or WEPR) is inferred from the coefficient keys.

        Parameter:
        pretrained_model_name_or_path: Path to the JSON file containing intercept and coefficients

        Returns:
        PretrainedLogisticRegression : instance of the class pre-configured with the loaded weights
        """
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        with pretrained_model_name_or_path.open("r") as f:
            weights = json.load(f)

        coeffs = weights["coefficients"]

        if "mean_entropy" in coeffs:  # epr
            coef_array = np.array([[coeffs["mean_entropy"]]], dtype=np.float32)  # (1, 1)
            n_features = 1
        else:  # wepr
            k = sum(1 for key in coeffs if key.startswith("mean_rank_"))
            mean_vals = [coeffs[f"mean_rank_{i}"] for i in range(1, k + 1)]
            max_vals = [coeffs[f"max_rank_{i}"] for i in range(1, k + 1)]
            coef_array = np.array([mean_vals + max_vals], dtype=np.float32)  # (1, 2k)
            n_features = 2 * k

        instance = cls()

        # set the attributes that .fit would normally set to trick sklearn into thinking the instance is fitted already
        instance.coef_ = coef_array
        instance.intercept_ = np.array([weights["intercept"]], dtype=np.float32)  # (1,)
        instance.classes_ = np.array([0, 1])  # by convention column 0 is calculated as 1-p and column 1 is p
        instance.n_features_in_ = n_features
        instance.n_iter_ = np.array([0])  # indicates that it took 0 iterations to get to the weights

        return instance
