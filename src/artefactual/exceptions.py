class ArtefactualWarning(UserWarning):
    """Base for all artefactual warnings (subclass of UserWarning → shown by default, filterable)."""


class EmptySequenceWarning(ArtefactualWarning):
    """A sequence had no tokens; scored at the classifier baseline."""


class ArtefactualError(Exception):
    """Base for all artefactual errors, so callers can catch the family."""


class UncalibratedModelError(ArtefactualError):
    """Raised when a detector requires a pretrained weight path for calibration.

    Deliberately not a silent fallback to an unfitted classifier: a detector that quietly
    trains on the caller's data would emit plausible-looking probabilities that no
    calibration backs, which is far harder to notice than an exception.
    """

    _MESSAGE = (
        "To enable this detector specify a `pretrained_model_name_or_path` — a model name "
        "from the registry or a path to a weights file. To fit your own calibration "
        "instead, pass `trainable=True` and call `fit`."
    )

    def __init__(self, message: str = _MESSAGE) -> None:
        super().__init__(message)
