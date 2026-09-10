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
