from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression

from artefactual.utils.io import load_calibration, load_weights

# Which registry a model *name* resolves through. A filesystem path bypasses both.
_LOADERS = {"weights": load_weights, "calibration": load_calibration}

Registry = Literal["weights", "calibration"]


class PretrainedLogisticRegression(LogisticRegression):
    # sklearn sets these inside fit; from_pretrained assigns them directly on a
    # never-fitted instance, so declare them for the type checker.
    coef_: np.ndarray
    intercept_: np.ndarray
    classes_: np.ndarray
    n_features_in_: int
    n_iter_: np.ndarray

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Path | str,
        *,
        registry: Registry = "weights",
        k: int | None = None,
    ) -> "PretrainedLogisticRegression":
        """
        Instantiate a LogisticRegression pre-loaded with calibrated coefficients.

        The metric (EPR or WEPR) is inferred from the coefficient keys, but which registry
        a bare *model name* resolves through cannot be — both registries are keyed by the
        same model names — so the caller states it. A filesystem path bypasses the registry.

        Args:
            pretrained_model_name_or_path: A model name in the chosen registry, or a path
                to a JSON file holding an intercept and coefficients.
            registry: "weights" for WEPR weights, "calibration" for an EPR calibration.
            k: The rank count the caller intends to score at. WEPR weights are validated
                against it, because their coefficient vector is fixed at the rank count
                used during calibration. `None` skips the check.

        Returns:
            An instance pre-configured with the loaded weights.

        Raises:
            ValueError: If the file is neither an EPR calibration nor WEPR weights, or if
                WEPR weights do not cover exactly `k` ranks.
        """
        # The two loaders declare different value types; both yield a coefficient mapping,
        # so pin it here rather than let the union leak into every lookup below.
        weights: dict[str, Any] = _LOADERS[registry](str(pretrained_model_name_or_path))
        coeffs: dict[str, float] = weights["coefficients"]

        if "mean_entropy" in coeffs:  # epr
            # A single pooled-entropy feature. The calibration records no rank count, so
            # `k` governs only how the rank axis is aligned upstream.
            coef_array = np.array([[coeffs["mean_entropy"]]], dtype=np.float32)  # (1, 1)
            n_features = 1
        elif "mean_rank_1" in coeffs:  # wepr
            # mean_rank_1 is present, so the count is at least 1
            file_k = sum(1 for key in coeffs if key.startswith("mean_rank_"))
            if k is not None and file_k != k:
                msg = (
                    f"Weights cover {file_k} rank(s) but k={k} was requested. WEPR "
                    f"coefficients are fixed at the rank count used during calibration; "
                    f"pass k={file_k}, or supply weights calibrated at k={k}."
                )
                raise ValueError(msg)
            mean_vals = [coeffs[f"mean_rank_{i}"] for i in range(1, file_k + 1)]
            max_vals = [coeffs[f"max_rank_{i}"] for i in range(1, file_k + 1)]
            coef_array = np.array([mean_vals + max_vals], dtype=np.float32)  # (1, 2k)
            n_features = 2 * file_k
        else:
            # Neither an EPR calibration nor WEPR weights; name what was found so the
            # reader can tell a malformed file from an unsupported metric.
            msg = (
                "Unrecognized weights format: expected an EPR calibration with a "
                "'mean_entropy' coefficient, or WEPR weights with 'mean_rank_1..k' and "
                f"'max_rank_1..k'. Got keys: {sorted(coeffs)}"
            )
            raise ValueError(msg)

        instance = cls()

        # set the attributes that .fit would normally set to trick sklearn into thinking the instance is fitted already
        instance.coef_ = coef_array
        instance.intercept_ = np.array([weights["intercept"]], dtype=np.float32)  # (1,)
        instance.classes_ = np.array([0, 1])  # by convention column 0 is calculated as 1-p and column 1 is p
        instance.n_features_in_ = n_features
        instance.n_iter_ = np.array([0])  # indicates that it took 0 iterations to get to the weights

        return instance
