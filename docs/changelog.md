# Changelog

Releases before the entries below are listed, with their notes, on the
[GitHub releases page](https://github.com/artefactory/artefactual/releases).

## Unreleased

### Breaking

- `predict_token_proba`, `EntropyTransformer.transform` and `transform_tokens` name their
  input `X` instead of `x`, as scikit-learn does. Positional calls are unaffected; a call
  that passes `x=` must pass `X=`.
- `fit` raises `ValueError` when the entropy step's reduction differs from the one the class
  fixes, for instance `EPR().set_params(entropy__reduction="wepr")`. Use `WEPR` instead.
- `EPR` and `WEPR` take responses only. A numeric table raises `TypeError` naming
  `EPRClassifier` and `WEPRClassifier`.

### Added

- `EPRClassifier` and `WEPRClassifier` score logprob arrays you already hold: a
  `(n_sequences, n_tokens, width)` array, or a numeric 2-D table read as one token per row.
  Targets must be binary. They pass scikit-learn's `check_estimator`, with one declared
  expected failure (`check_estimators_overwrite_params`, which scikit-learn lists for
  `Pipeline` itself). `EPR` and `WEPR` are the same estimators with a `LogProbParser` in
  front. See {doc}`guide/sklearn`.
- `BaseEntropyClassifier`, the pipeline the four estimators share, and the class-level
  `reduction`, `n_features(k)` and `k_for(n_features)`.
- `EntropyTransformer` keeps the first `k` ranks of input that is wider, which are the most
  likely candidates, so a `GridSearchCV` over `k` works on one wide array. Input narrower
  than `k` is used as it is.
- Feature names: `get_feature_names_out` works through the pipeline, and scikit-learn's
  `set_output` (`"pandas"` or `"polars"`) gives named DataFrames on slices such as
  `detector[:-1]`. `EntropyTransformer` takes an optional `k`, which the estimators set on it.
  Rank columns are `rank0`..`rank{k-1}`; `EPR` and `EPRClassifier` have one feature, `epr`;
  `WEPR` and `WEPRClassifier` have `wepr_mean_rank*` then `wepr_max_rank*`. A classifier
  fitted on a table with column names uses them in place of `rank*`.
- `from artefactual import EPR, WEPR, EPRClassifier, WEPRClassifier`: the four estimators are
  exported from the package root.
- numpydoc docstrings on the estimators, documenting the fitted attributes (`classes_`,
  `n_features_in_`, `named_steps`, `steps`, `estimator`).
- Guide pages: {doc}`install`, {doc}`guide/sklearn`, this changelog, and a curated
  {doc}`api` reference. `CITATION.cff` at the repository root.

### Changed

- `predict_proba` columns follow `classes_`. With `0`/`1` labels column 1 is the
  hallucination class; with other labels use `list(detector.classes_).index(label)`.
- `get_params` reports `estimator=None` while the final step is the untouched default, so
  `clone` rebuilds an equal detector.
- The estimator tags declare a binary classifier. The classifiers declare 2-D input.
