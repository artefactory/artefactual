"""Warnings and errors raised across the package.

Both hierarchies have a base class so callers can catch or filter the whole family:
`ArtefactualWarning` for recoverable conditions, `ArtefactualError` for failures.
"""


class ArtefactualWarning(UserWarning):
    """Base for all artefactual warnings (subclass of UserWarning → shown by default, filterable)."""


class EmptySequenceWarning(ArtefactualWarning):
    """A sequence had no tokens; scored at the classifier baseline."""


class ArtefactualError(Exception):
    """Base for all artefactual errors, so callers can catch the family."""


class UncalibratedModelError(ArtefactualError):
    """Raised when a detector is built with neither pretrained weights nor `trainable=True`.

    A detector has no default calibration: the coefficients are model-specific, so an
    unfitted classifier would return probabilities backed by nothing. The two ways to
    resolve it are naming a calibration or asking for an unfitted pipeline to `fit`.
    """

    _MESSAGE = (
        "To enable this detector specify a `pretrained_model_name_or_path` — a model name "
        "from the registry or a path to a weights file. To fit your own calibration "
        "instead, pass `trainable=True` and call `fit`."
    )

    def __init__(self, message: str = _MESSAGE) -> None:
        super().__init__(message)
