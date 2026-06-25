class ArtefactualWarning(UserWarning):
    """Base for all artefactual warnings (subclass of UserWarning → shown by default, filterable)."""


class EmptySequenceWarning(ArtefactualWarning):
    """A sequence had no tokens; scored at the classifier baseline."""
