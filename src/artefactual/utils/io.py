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

import skops.io as sio
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

MODEL_FILENAME = "model.skops"


def resolve_estimator(identifier: str | Path) -> Path:
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

    # A directory is accepted and read as the `model.skops` inside it, which is the
    # layout a published repository has once downloaded, so a clone and a repository name
    # each the same file.
    local = Path(identifier)
    if local.is_file():
        return local
    if local.is_dir() and (local / MODEL_FILENAME).is_file():
        return local / MODEL_FILENAME

    # Imported here rather than at module scope: loading a local file never reaches the
    # Hub, so it does not pay the import cost. The dependency is required either way.
    from huggingface_hub import hf_hub_download

    repo_id = str(identifier)
    try:
        return Path(hf_hub_download(repo_id, MODEL_FILENAME))
    except Exception as error:
        msg = (
            f"Could not load an estimator from '{identifier}'. Expected a path to a "
            f".skops file, a directory holding {MODEL_FILENAME}, or a Hugging Face "
            f"repository id holding one. Fetching it failed with: {error}"
        )
        raise ValueError(msg) from error


def load_estimator(identifier: str | Path, trusted: list[str] | None = None) -> BaseEstimator:
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
    path = resolve_estimator(identifier)
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


def dump_estimator(estimator: BaseEstimator, path: str | Path) -> str | Path:
    check_is_fitted(estimator)
    path = Path(path)
    if path.is_dir():
        path = path / MODEL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(estimator, path)
    return path
