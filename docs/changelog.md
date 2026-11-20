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
- `artefactual.preprocessing.openai_parser` is removed: call `top_logprobs()` and
  `sampled_logprobs()` on `ChatCompletion` or `ResponsesPayload`.

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
- `from_pretrained` (and `resolve_estimator`, `load_estimator`) take keyword-only `revision`,
  `token` and `cache_dir`, forwarded to the Hugging Face download, so a published detector
  can be pinned to a commit.
- `EntropyTransformer` validates its `reduction` and `k` at `fit`, and a detector validates `k`
  and `estimator`, with scikit-learn's `InvalidParameterError` (a `ValueError`).
- `from artefactual import EPR, WEPR, EPRClassifier, WEPRClassifier`: the four estimators are
  exported from the package root.
- numpydoc docstrings on the estimators, documenting the fitted attributes (`classes_`,
  `n_features_in_`, `named_steps`, `steps`, `estimator`).
- Guide pages: {doc}`install`, {doc}`guide/sklearn`, this changelog, and a curated
  {doc}`api` reference. `CITATION.cff` at the repository root.

- `Judgment`, the schema of a judge's reply (`judgment`, `explanation`), usable as a
  structured-output `response_format`. `Judgment.from_text`, `from_completion` and
  `from_batch` read a reply, a completion or a whole batch of verdicts, keeping the
  explanation; `read_judgment` keeps working and returns the same verdicts.
- `ChatCompletion.top_logprobs()` and `sampled_logprobs()`, and the same two on the Responses
  payload model, share one implementation through `LogprobsMixin`. A class that implements
  the mixin is registered as a format, so a new format is one class. `LogprobsMixin.parse`
  validates a completion, a Responses payload or a batch line into the model it is;
  `ChatCompletion.parse` and `ResponsesPayload.parse` accept their own format only.
  `BatchRequestOutput.payload()` returns the body of a successful line. `ChatCompletion`,
  `ResponsesPayload` and `BatchRequestOutput` import from `artefactual` directly;
  `LogprobsMixin` stays in `artefactual.preprocessing.response_models`. `TokenLogprobs.candidates()`
  gives a token's candidates, highest first. `parse_top_logprobs`, `parse_sampled_token_logprobs`
  and `LogProbParser` return the same results and raise the same errors as before.

- `epr_reduction` and `wepr_reduction`, the two reductions behind the detectors, and
  `EntropyTransformer.validate`, its input check, are public, and import from
  `artefactual.scoring`.

### Changed

- `predict_proba` columns follow `classes_`. With `0`/`1` labels column 1 is the
  hallucination class; with other labels use `list(detector.classes_).index(label)`.
- `get_params` reports `estimator=None` while the final step is the untouched default, so
  `clone` rebuilds an equal detector.
- The estimator tags declare a binary classifier. The classifiers declare 2-D input.

### Fixed

- `detector[::2]` raises `ValueError`, as `Pipeline` does, instead of returning a pipeline of
  an arbitrary subset of steps.
- `EntropyTransformer.transform` and `transform_tokens` accept a nested list and raise
  `ValueError` on infinite logprobs, rather than a `beartype` error or an infinite feature.
- `from_pretrained` refuses an estimator that is not a fitted two-class classifier, with a
  `ValueError` naming the file.
- A repository that cannot be found, a missing revision and a malformed repository id raise
  `ValueError`; connection and authentication errors now propagate as the
  `huggingface_hub` error they are, instead of being wrapped.
