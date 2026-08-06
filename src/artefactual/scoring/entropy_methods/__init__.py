"""The uncertainty measurement behind the shipped detectors.

An implementation detail of `epr()` and `wepr()`; callers building a detector do not need
anything from here. `EntropyTransformer` and `EntropyContributionsMixin` are supported
under `artefactual.scoring`; within the package they are imported from the modules that
define them.
"""
