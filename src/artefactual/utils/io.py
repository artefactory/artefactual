"""Persistence of estimators: skops on disk, Hugging Face repositories by name.

An estimator is named by a Hugging Face repository id, a `.skops` file, or a directory
holding one. Nothing here maps a language model to a detector: which detector suits which
model is documentation, so publishing one costs a README line rather than a release.

The persisted artifact is the fitted estimator alone, not a whole detector. The pipeline
around it -- parsing, entropy reduction -- is assembled from this package at the version
installed, so a published estimator carries no reference to this package's modules and
survives their renaming.
"""

from pathlib import Path
from typing import Any

import skops.io as sio
from beartype import beartype
from beartype.typing import Protocol
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

MODEL_FILENAME = "model.skops"
"""Name of the estimator file inside a directory or a published repository.

A module constant rather than a mixin attribute: the loading steps and the saving step
share it, and the owner protocols below are meant to hold methods only.
"""


def _require_k(kwargs: dict[str, Any]) -> None:
    """Refuse constructor options that omit `k`, naming the remedy.

    `k` is the candidates-per-token width the weights were trained at and no default is
    safe, but it travels inside `**kwargs`, where its absence would surface as a bare
    missing-argument error after a download.

    Raises
    ------
    TypeError
        If `"k"` is not a key of `kwargs`.
    """
    if "k" not in kwargs:
        msg = (
            "`k` is required: pass the k the model was trained with, as the weights are "
            "fixed at that width (the published artefactory detectors use k=15; see the "
            "README table)."
        )
        raise TypeError(msg)


class EstimatorFactory(Protocol):
    """What `PretrainedMixin.from_pretrained` requires of its class: the three steps it takes.

    `from_pretrained` finds a file (`cached_file`), reads it (`load_file`) and builds the
    instance (`from_estimator`). `PretrainedMixin` supplies the first two; a class supplies
    the third by assembling itself around a loaded estimator and rejecting one that does not
    fit. A class that overrides either of the first two changes where weights come from or
    how they are read, and `from_pretrained` follows.

    A protocol of its own, apart from `EstimatorOwner`, because a protocol with a data
    member (a property is one) cannot be checked with `issubclass`, which is how a class
    is checked.
    """

    @classmethod
    def check_options(cls, kwargs: dict[str, Any]) -> None:
        """Refuse `from_pretrained`'s extra options before any file is looked up."""

    @classmethod
    def from_estimator(cls, estimator: BaseEstimator, identifier: str | Path, **kwargs: Any) -> Any:
        """Wrap a loaded estimator in the object that scores with it.

        Args:
            estimator: The fitted estimator read from the file.
            identifier: What named it, so a rejection can name the file it came from.
            **kwargs: Whatever `from_pretrained` was called with beyond its own options.

        Returns:
            An instance of the class, ready to score.

        Raises:
            ValueError: If the estimator does not fit the class.
        """

    @classmethod
    def cached_file(
        cls,
        identifier: str | Path,
        *,
        revision: str | None,
        token: str | bool | None,
        cache_dir: str | Path | None,
    ) -> Path:
        """The local `.skops` file *identifier* names. See `PretrainedMixin.cached_file`."""

    @classmethod
    def load_file(cls, path: str | Path, trusted: list[str] | None = None) -> BaseEstimator:
        """The estimator a `.skops` file holds. See `PretrainedMixin.load_file`."""


class EstimatorOwner(Protocol):
    """What `SavePretrainedMixin.save_pretrained` requires of its object: a fitted estimator.

    Checked on the instance. Satisfying it evaluates `estimator`, so a property that
    raises raises at the check, before `save_pretrained` runs.
    """

    @property
    def estimator(self) -> BaseEstimator:
        """The estimator this object scores with."""


class PretrainedMixin:
    """Loading an instance from a published estimator: `from_pretrained`.

    The class using it must satisfy `EstimatorFactory`. It loads the same way whether the
    caller names a file, a directory or a published model.

    `cached_file` and `load_file` are public override points: the first decides where the
    weights come from, the second how they are read. A subclass overrides one to load from
    somewhere else (another registry, an encrypted store) or to read another format, and
    `from_pretrained` uses the override without further change.
    """

    @classmethod
    def cached_file(
        cls,
        identifier: str | Path,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
    ) -> Path:
        """The local `.skops` file *identifier* names, downloading it if it names a repository.

        A `.skops` file is returned as it is. A directory resolves to the `MODEL_FILENAME`
        inside it, the layout a downloaded repository has, so a clone and a repository name
        reach the same file. Anything else is a Hugging Face repository id, downloaded into
        the cache. The only step that reaches the network is that download; which file to
        read is decided beforehand.

        Override to change where weights come from.

        Args:
            identifier: A Hugging Face repository id, a `.skops` file, or a directory
                holding `MODEL_FILENAME`.
            revision: Branch, tag or commit of the repository to download. `None` takes the
                default branch, which a published repository may move; pin a commit for a
                reproducible download. Ignored for a local path.
            token: Hugging Face access token, or `True` to use the locally saved one. `None`
                uses the library's default credentials. Ignored for a local path.
            cache_dir: Where to cache the download. `None` uses the Hugging Face cache.
                Ignored for a local path.

        Returns:
            Path to a local `.skops` file. For a repository the path is in the Hugging Face
            cache, so a second call for the same revision does not download again.

        Raises:
            ValueError: If *identifier* names no local file and no repository, revision or
                file that the Hub knows (or, offline, none already in the cache).
                Connection, authentication and rate-limit failures are not wrapped: they
                propagate as the `huggingface_hub` error they are.
        """
        local = Path(identifier)
        if local.is_file():
            return local
        if local.is_dir() and (local / MODEL_FILENAME).is_file():
            return local / MODEL_FILENAME

        # Imported here rather than at module scope: loading a local file never reaches the
        # Hub, so it does not pay the import cost. The dependency is required either way.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import (
            EntryNotFoundError,
            HFValidationError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        try:
            return Path(
                hf_hub_download(str(identifier), MODEL_FILENAME, revision=revision, token=token, cache_dir=cache_dir)
            )
        except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError, HFValidationError) as error:
            msg = (
                f"Could not load an estimator from '{identifier}'. Expected a path to a "
                f".skops file, a directory holding {MODEL_FILENAME}, or a Hugging Face "
                f"repository id holding one. Fetching it failed with: {error}"
            )
            raise ValueError(msg) from error

    @classmethod
    def load_file(cls, path: str | Path, trusted: list[str] | None = None) -> BaseEstimator:
        """Read a fitted estimator from a `.skops` file, refusing types not asked for.

        Never loads with blanket trust. skops names every type in the file it does not
        already trust, and those are reported rather than accepted: an estimator is data
        from somewhere else, and loading one must not become a way to have it construct
        arbitrary objects. The published estimators hold a plain `LogisticRegression`,
        which skops trusts on its own, so they need nothing here.

        Args:
            path: A `.skops` file.
            trusted: Type names to accept beyond the ones skops trusts by default. Needed
                only for an estimator fitted with a `classifier=` this package did not
                write.

        Returns:
            The estimator the file holds.

        Raises:
            ValueError: If the file holds a type that is neither trusted by default nor
                listed in *trusted*.

        Override to change how weights are read; keep the refusal of types not asked for.
        """
        # Checked here rather than left to `sio.load`: its refusal is a `TypeError` that
        # lists the types but does not say how to accept them, and callers catch the
        # `ValueError` this raises.
        unknown = [name for name in sio.get_untrusted_types(file=path) if name not in (trusted or [])]
        if unknown:
            msg = (
                f"The estimator at '{path}' holds types this package does not load by "
                f"default: {unknown}. Loading a file constructs the objects it names, so "
                f"they are refused unless asked for. Pass trusted={unknown!r} if this file "
                f"is yours and those types are what you saved."
            )
            raise ValueError(msg)
        return sio.load(path, trusted=trusted)

    @classmethod
    def check_options(cls, kwargs: dict[str, Any]) -> None:
        """Refuse `from_pretrained`'s extra options before any file is looked up.

        Called with the keyword options destined for `from_estimator`, ahead of path
        resolution and download, so a mistake costs no I/O. Accepts everything by default;
        override to reject options the class cannot build from.
        """

    @classmethod
    @beartype
    def from_pretrained(
        cls: type[EstimatorFactory],
        identifier: str | Path,
        *,
        trusted: list[str] | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """An instance carrying published weights, ready to score.

        The route to a fitted object that never calls `fit`: the weights were fitted
        elsewhere, and what is rebuilt here is only the machinery around them. The class
        must define `from_estimator` (`EstimatorFactory`); one that does not is rejected
        before anything is fetched.

        Parameters
        ----------
        identifier : str or Path
            A Hugging Face repository id, a `.skops` file, or a directory holding
            `MODEL_FILENAME`.
        trusted : list of str, default=None
            Type names to accept beyond skops' defaults. See `load_file`.
        revision : str, default=None
            Branch, tag or commit of the Hugging Face repository to download; `None` takes
            the default branch. Pin a commit for a reproducible load. Ignored for a local
            path.
        token : str or bool, default=None
            Hugging Face access token, for a private repository. Ignored for a local path.
        cache_dir : str or Path, default=None
            Download cache directory; `None` uses the Hugging Face cache. Ignored for a
            local path.
        **kwargs
            Passed to `from_estimator`, which decides what the class needs. `k`, the
            candidates per token the weights were trained at, is required.

        Returns
        -------
        object
            An instance of the class, carrying the loaded estimator.

        Raises
        ------
        TypeError
            If a class that requires an option (`k` for the detectors) was not given it;
            checked before anything is resolved or fetched.
        ValueError
            If the identifier resolves to nothing, the file holds a type that was not
            asked for, or the estimator does not fit the class.

        Examples
        --------
        >>> detector = WEPR.from_pretrained("artefactory/wepr-phi4", k=15)  # doctest: +SKIP
        >>> detector = WEPR.from_pretrained("/path/to/model.skops", k=15)  # doctest: +SKIP
        >>> detector = WEPR.from_pretrained(
        ...     "artefactory/wepr-phi4", k=15, revision="main"
        ... )  # doctest: +SKIP
        """
        cls.check_options(kwargs)
        path = cls.cached_file(identifier, revision=revision, token=token, cache_dir=cache_dir)
        return cls.from_estimator(cls.load_file(path, trusted=trusted), identifier, **kwargs)


class SavePretrainedMixin:
    """Writing the fitted estimator of an object: `save_pretrained`.

    The object using it must satisfy `EstimatorOwner`. It is usable without `PretrainedMixin`,
    so an owner that only ships weights does not inherit a method it cannot support.
    """

    @beartype
    def save_pretrained(self: EstimatorOwner, path: str | Path) -> Path:
        """Write this object's fitted estimator to a `.skops` file.

        Persists the estimator alone rather than the whole object, so the file stays
        readable by a differently-versioned install: it names only scikit-learn types.

        Parameters
        ----------
        path : str or Path
            Destination file, or a directory in which to write `MODEL_FILENAME`. Missing
            parent directories are created.

        Returns
        -------
        Path
            The path written.

        Raises
        ------
        beartype.roar.BeartypeCallHintParamViolation
            If the object has no `estimator` (see `EstimatorOwner`).
        NotFittedError
            If the estimator has not been fitted. An unfitted estimator writes a file that
            looks valid and fails only when something later tries to score with it, far
            from the call that forgot to fit.

        Examples
        --------
        >>> detector.save_pretrained("my_detector.skops")  # doctest: +SKIP
        >>> WEPR.from_pretrained("my_detector.skops", k=15)  # doctest: +SKIP
        """
        check_is_fitted(self.estimator)
        path = Path(path)
        if path.is_dir():
            path = path / MODEL_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(self.estimator, path)
        return path
