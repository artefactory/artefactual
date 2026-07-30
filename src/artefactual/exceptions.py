class ArtefactualWarning(UserWarning):
    """Base for all artefactual warnings (subclass of UserWarning → shown by default, filterable)."""


class EmptySequenceWarning(ArtefactualWarning):
    """A sequence had no tokens; scored at the classifier baseline."""


class ArtefactualError(Exception):
    """Base for all artefactual errors, so callers can catch the family."""


class UncalibratedModelError(ArtefactualError):
    """Raised when a detector requires a pretrained weight path for calibration."""

    def __init__(self, message: str = "To enable this detector specify a `pretrained_model_name_or_path`.") -> None:
        super().__init__(message)
