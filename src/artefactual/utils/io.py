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
from typing import Any, Literal

import skops.io as sio
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

Reduction = Literal["epr", "wepr"]


class EstimatorPersistenceMixin:
    """Reading and writing estimators, for the estimator that owns one.

    Mixed into the detector so persistence is part of its surface rather than a separate
    set of helpers: a detector saves the fitted estimator it scores with, and loads one
    the same way regardless of whether the caller names a file or a published model.
    """

    MODEL_FILENAME = "model.skops"

    @classmethod
    def local_estimator(cls, identifier: str | Path) -> Path | None:
        """The `.skops` file *identifier* names on disk, or None if it names none.

        A directory is accepted and read as the `model.skops` inside it, which is the
        layout a published repository has once downloaded, so a clone and a repository name
        reach the same file.
        """
        local = Path(identifier)
        if local.is_file():
            return local
        if local.is_dir() and (local / cls.MODEL_FILENAME).is_file():
            return local / cls.MODEL_FILENAME
        return None

    @classmethod
    def resolve_estimator(cls, identifier: str | Path) -> Path:
        """Resolve *identifier* to a local `.skops` file, downloading it if it names a repo.

        The only step here that reaches the network is the download itself; which file to
        read, and which repository a name means, are both decided beforehand.

        Args:
            identifier: A Hugging Face repository id, a `.skops` file, or a directory
                holding `model.skops`.

        Returns:
            Path to a local `.skops` file. For a repository the path is in the Hugging Face
            cache, so a second call for the same revision does not download again.

        Raises:
            ValueError: If *identifier* names no local file and no repository that could be
                fetched.
        """
        local = cls.local_estimator(identifier)
        if local is not None:
            return local

        # Imported here rather than at module scope: loading a local file never reaches the
        # Hub, so it does not pay the import cost. The dependency is required either way.
        from huggingface_hub import hf_hub_download

        repo_id = str(identifier)
        try:
            return Path(hf_hub_download(repo_id, cls.MODEL_FILENAME))
        except Exception as error:
            msg = (
                f"Could not load an estimator from '{identifier}'. Expected a path to a "
                f".skops file, a directory holding {cls.MODEL_FILENAME}, or a Hugging Face "
                f"repository id holding one. Fetching it failed with: {error}"
            )
            raise ValueError(msg) from error

    @classmethod
    def read_estimator(cls, path: str | Path, trusted: list[str] | None = None) -> BaseEstimator:
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
        """
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
    def load_estimator(cls, identifier: str | Path, trusted: list[str] | None = None) -> BaseEstimator:
        """Resolve *identifier* and read the estimator it names.

        Args:
            identifier: A repository id or path. See `resolve_estimator`.
            trusted: Type names to accept beyond skops' defaults.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If the identifier resolves to nothing, or the file holds a type
                that was not asked for.
        """
        return cls.read_estimator(cls.resolve_estimator(identifier), trusted=trusted)

    def save_estimator(self, path: str | Path) -> Path:
        """Write this object's fitted estimator to a `.skops` file.

        Persists the estimator alone rather than the whole object, so the file stays
        readable by a differently-versioned install: it names only scikit-learn types.

        Args:
            path: Destination file, or a directory in which to write `model.skops`. Missing
                parent directories are created.

        Returns:
            The path written.

        Raises:
            NotFittedError: If the estimator has not been fitted. An unfitted estimator
                writes a file that looks valid and fails only when something later tries to
                score with it, far from the call that forgot to fit.
        """
        check_is_fitted(self.estimator)
        path = Path(path)
        if path.is_dir():
            path = path / self.MODEL_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        sio.dump(self.estimator, path)
        return path

    @property
    def estimator(self) -> Any:
        """The fitted estimator this object scores with. Overridden by the owner."""
        raise NotImplementedError
