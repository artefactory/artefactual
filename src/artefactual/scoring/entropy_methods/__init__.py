"""The uncertainty measurement behind the shipped detectors.

An implementation detail of `EPR` and `WEPR`; callers building a detector do not need
anything from here. `EntropyTransformer`, `EntropyContributionsMixin` and the poolings
`Pooling`, `EPRPooling`, `WEPRPooling`, `epr_reduction` and `wepr_reduction` are supported
under `artefactual.scoring`;
within the package they are imported from the modules that define them.
"""
