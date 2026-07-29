import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from artefactual.utils.io import load_weights


class PretrainedLogisticRegression(LogisticRegression):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Path | str) -> "PretrainedLogisticRegression":
        """
        Instantiate and initialize a LogisticRegression model using pre-trained weights.
        The metric (EPR or WEPR) is inferred from the coefficient keys.

        Args:
        pretrained_model_name_or_path: Path to the JSON file containing intercept and coefficients

        Returns:
        PretrainedLogisticRegression : instance of the class pre-configured with the loaded weights
        """

        if pretrained_model_name_or_path is None:
            return warnings.warn(
                "EPR is currently not calibrated. "
                "To enable calibration, please specify a `pretrained_model_name_or_path`.",
                UserWarning,
                stacklevel=2,
            )

        weights = load_weights(pretrained_model_name_or_path)

        coeffs = weights["coefficients"]

        if "mean_entropy" in coeffs:  # epr
            coef_array = np.array([[coeffs["mean_entropy"]]], dtype=np.float32)  # (1, 1)
            n_features = 1
        elif "mean_rank_1" in coeffs:  # wepr
            k = sum(1 for key in coeffs if key.startswith("mean_rank_"))
            if k == 0:
                msg = "Unrecognized coefficient keys"
                raise ValueError(msg)
            mean_vals = [coeffs[f"mean_rank_{i}"] for i in range(1, k + 1)]
            max_vals = [coeffs[f"max_rank_{i}"] for i in range(1, k + 1)]
            coef_array = np.array([mean_vals + max_vals], dtype=np.float32)  # (1, 2k)
            n_features = 2 * k
        else:
            msg = "Unrecognized coefficient keys"
            raise ValueError(msg)

        instance = cls()

        # set the attributes that .fit would normally set to trick sklearn into thinking the instance is fitted already
        instance.coef_ = coef_array
        instance.intercept_ = np.array([weights["intercept"]], dtype=np.float32)  # (1,)
        instance.classes_ = np.array([0, 1])  # by convention column 0 is calculated as 1-p and column 1 is p
        instance.n_features_in_ = n_features
        instance.n_iter_ = np.array([0])  # indicates that it took 0 iterations to get to the weights

        return instance
