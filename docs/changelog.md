# Changelog

Releases before the entries below are listed, with their notes, on the
[GitHub releases page](https://github.com/artefactory/artefactual/releases).

## Unreleased

Changes against 2026.11.1.

### Breaking

- `EPRClassifier` is now `EPR` and `WEPRClassifier` is now `WEPR`. The 2026.11.1 `EPR` and
  `WEPR`, which took responses, `PoolingDetector`, `BaseDetector` and `BaseEntropyClassifier` are
  removed.
- `EPR` and `WEPR` take logprob arrays. Passing responses raises a `TypeError` naming
  `is_logprob_array`: parse first with `LogProbParser(k=15).transform(responses)`, or use
  `make_detector(k=15)`.
- A `Pooling` subclass must be a `@dataclasses.dataclass(frozen=True)` with its parameters as
  fields; an undecorated subclass raises `TypeError` when instantiated. An array field is
  declared `dataclasses.field(compare=False)`. A pooling cannot be mutated after construction.
- `FunctionPooling` takes no `name=`: it is named after the function, and a lambda or a
  `partial` is named `pooling`. Define a named function to name it.
- `EntropyTransformer` has no `k`: it reads the width from the data, and a fitted one refuses
  another width.
- `WEPR` has no `k`, and `WEPR.from_pretrained(repo)` takes none.
- `EPR(k)` pins the input width. An array of another width is refused with
  `trained on 15 candidates per token, got 10: use LogProbParser(k=15)`, for a loaded or cloned
  `EPR` too. `EPR.from_pretrained(repo, k=15)` still requires `k`.
- `scripts/train_detector.py --pooling` takes `epr` or `wepr` only.

### Added

- `make_detector(*, k, classifier=None)` returns a `DetectorPipeline` (`LogProbParser(k=k)` then
  `classifier`, `WEPR()` by default): a `Pipeline` with `predict_token_proba`, from
  responses to scores.
- `token_proba(steps, X)`: per-token probabilities through a list of pipeline steps. Each step
  runs `transform_tokens` when it has one, else `transform`, and the last step's
  `predict_proba` scores each token, with `NaN` at padding. `DetectorPipeline`
  uses it for `predict_token_proba`.
- `TokenProbaMixin`: `predict_token_proba` for a classifier that builds a fitted `pipeline_`,
  through `token_proba`. `EPR`, `WEPR` and a detector of your own use it.
- The runtime-checkable protocols `TokenTransformer` and `TokenClassifier`.
  `make_detector` refuses a `classifier` without `predict_token_proba`, and `DetectorPipeline`
  refuses at `fit` a layout with no per-token step, with a `TypeError`.
- `k` is searchable through the parser: `{"logprobparser__k": [5, 10, 15]}`.
- The scikit-learn guide shows how to write your own detector and the advanced `make_pipeline`
  recipes (end to end on responses, `GridSearchCV`, `FeatureUnion`).

### Changed

- `from_pretrained` returns a detector backed by a `FrozenEstimator`. It predicts, also after
  `clone`, and a `fit` keeps the loaded weights.
- `check_estimator` runs on `EPR`, `WEPR`, `EntropyTransformer` and a custom detector with no
  expected failure.
- `EntropyTransformer.transform_tokens` calls the pooling on real tokens only, and returns
  `NaN` rows at padding tokens, so a pooling built on `np.nanmedian` or `np.nanmean` no longer
  warns about all-`NaN` padding windows.
- The docs and the example notebooks use `EPR`, `WEPR` and `make_detector` for basic usage, and
  `make_pipeline` only in the advanced section.

### Fixed

- `LogProbParser` validates `k` at `fit` with scikit-learn's `InvalidParameterError`.

## 2026.11.0

### Breaking

- `predict_token_proba`, `EntropyTransformer.transform` and `transform_tokens` name their
  input `X` instead of `x`, as scikit-learn does. Positional calls are unaffected; a call
  that passes `x=` must pass `X=`.
- `fit` raises `ValueError` when the entropy step's pooling differs from the one the class
  fixes, for instance `EPRClassifier(k=15).set_params(entropy__pooling=WEPRPooling())`. Use
  `WEPRClassifier` instead.
- `EPR` and `WEPR` take responses only. A numeric table raises `TypeError` naming
  `EPRClassifier` and `WEPRClassifier`.
- `artefactual.preprocessing.openai_parser` is removed: call `top_logprobs()` and
  `sampled_logprobs()` on `ChatCompletion` or `ResponsesPayload`.
- Estimator persistence is split into `PretrainedMixin` (`from_pretrained`, with the public
  override points `cached_file` and `load_file`) and `SavePretrainedMixin` (`save_pretrained`),
  importable from `artefactual.utils.io` only. Renamed:
  `EstimatorPersistenceMixin` to `PretrainedMixin` and `SavePretrainedMixin`;
  `save_estimator` to `save_pretrained`; `resolve_estimator` and `local_estimator` to
  `cached_file`; `read_estimator` to `load_file`; `_from_estimator` to `from_estimator`.
  `load_estimator` is removed: call `from_pretrained`.
- `EntropyTransformer(reduction=...)` is now `EntropyTransformer(pooling=...)`, and the entropy
  step's parameter is `entropy__pooling`. A pooling is a `Pooling` instance (`EPRPooling()`,
  `WEPRPooling()`, or a subclass of `Pooling` of your own) or a plain function
  `f(x, axis)`; a string is refused with a `TypeError`. There is no name registry: nothing
  is selected by name, and no global state holds a pooling. `STRATEGIES` and
  `artefactual.utils.io.Reduction` are removed, and so is `BaseDetector.trainable`: construct
  `EPR`, `WEPR` or `PoolingDetector(pooling)`. The class attribute `reduction` of the
  estimators is now `pooling`, a `Pooling` instance, and `EntropyTransformer.reduction_fn` is
  gone: call `transformer.applied_pooling()`.
- `scripts/train_detector.py --reduction` is now `--pooling`: `epr`, `wepr`, or an import path
  `package.module:Attr` to a `Pooling` subclass, a `Pooling` instance or a function.
- `EntropyTransformer(pooling)` and `PoolingDetector.from_pretrained(path, pooling=...)` require
  the pooling: nothing defaults to `EPRPooling()` and `pooling=None` is refused. `EPR`, `WEPR`
  and their classifiers still fix theirs.
- `k` has no default. `EPR`, `WEPR`, `EPRClassifier`, `WEPRClassifier`, `PoolingDetector` and
  `EntropyTransformer` require it, `artefactual.scoring.base_detector.DEFAULT_K` is removed, and
  `scripts/train_detector.py` requires `--k`. `EntropyTransformer(pooling, k=None)` still keeps
  every rank. Pass the `k` the weights were trained at, which is 15 for the published detectors.
- `get_feature_names_out` of a custom pooling function returns names (its own name, or
  `name_0`, ...) instead of raising `ValueError`.
- A fitted `EntropyTransformer` refuses at `transform` a cube whose last dimension differs
  from the width it was fitted on, with a `ValueError`, instead of scoring it silently. Fit
  and transform on the same width.
- `Pooling.__call__` takes `X`, as the estimator methods do, where it took `x`: an override
  that renames the argument still works when it is called positionally, which
  `EntropyTransformer` does.

### Added

- `Pooling`, `EPRPooling`, `WEPRPooling` and `FunctionPooling`: subclass `Pooling`, implement
  `__call__(X, axis)`, and pass an instance to `EntropyTransformer` or `PoolingDetector`.
  Override `name`, `width(k)`, `constant_width` and `feature_names(ranks)` to name the
  features or declare the width; otherwise the width is probed and a one-feature pooling is
  named after the class. A bare function works too and is named after itself.
  `PoolingDetector(pooling, k, estimator)` is the detector over any pooling; its file holds the
  classifier alone, so `PoolingDetector.from_pretrained` takes `pooling=`.
  `BaseDetector` and `BaseEntropyClassifier` are importable from the package root.
- `EPRClassifier` and `WEPRClassifier` score logprob arrays you already hold: a
  `(n_sequences, n_tokens, width)` array, or a numeric 2-D table read as one token per row.
  Targets must be binary. They pass scikit-learn's `check_estimator`, with one declared
  expected failure (`check_estimators_overwrite_params`, which scikit-learn lists for
  `Pipeline` itself); `EntropyTransformer` passes it too, and the checks that need no data
  are run by hand on `LogProbParser`. `EPR` and `WEPR` are the same estimators with a
  `LogProbParser` in front, so they take responses and are not run through
  `parametrize_with_checks`. See {doc}`guide/sklearn`.
- `BaseEntropyClassifier`, the pipeline the four estimators share, and the class-level
  `pooling`.
- `EntropyTransformer` keeps the first `k` ranks of input that is wider, which are the most
  likely candidates, so a `GridSearchCV` over `k` works on one wide array. Input narrower
  than `k` is used as it is.
- Feature names: `get_feature_names_out` works through the pipeline, and scikit-learn's
  `set_output` (`"pandas"` or `"polars"`) gives named DataFrames on slices such as
  `detector[:-1]`. `EntropyTransformer` takes an optional `k`, which the estimators set on it.
  Rank columns are `rank0`..`rank{k-1}`; `EPR` and `EPRClassifier` have one feature, `epr`;
  `WEPR` and `WEPRClassifier` have `wepr_mean_rank*` then `wepr_max_rank*`. A classifier
  fitted on a table with column names uses them in place of `rank*`.
- `from_pretrained` takes keyword-only `revision`, `token` and `cache_dir`, forwarded to the
  Hugging Face download, so a published detector can be pinned to a commit.
- `EntropyTransformer` validates its `pooling` and `k` at `fit`, and a detector validates `k`
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
  `BatchRequestOutput.payload()` returns the body of a successful line. `TokenLogprobs.candidates()`
  gives a token's candidates, highest first. `parse_top_logprobs`, `parse_sampled_token_logprobs`
  and `LogProbParser` return the same results and raise the same errors as before.

- `epr_reduction` and `wepr_reduction`, the two reductions behind the detectors, and
  `EntropyTransformer.validate`, its input check, are public. Together with `EPR`, `WEPR`,
  `EPRClassifier`, `WEPRClassifier`, `EntropyTransformer`, `LogProbParser`, `Judgment`,
  `ChatCompletion`, `ResponsesPayload`, `BatchRequestOutput` and `read_batch`, they import
  from `artefactual` directly. `LogprobsMixin` stays in `artefactual.preprocessing.response_models`.

### Changed

- `EntropyTransformer` names `min(k, n_features_in_)` features once fitted, which is the
  width it hands on when the input is narrower than `k`.
- `predict_token_proba` on an empty batch returns an array of shape `(0, 0, 1)`, and on a
  batch whose tokens are all padding returns `NaN`, instead of raising.
- `predict` after the pooling was swapped on a fitted detector raises a `ValueError` asking
  for a refit, instead of scoring with coefficients fitted for the old pooling.
- `predict_proba` columns follow `classes_`. With `0`/`1` labels column 1 is the
  hallucination class; with other labels use `list(detector.classes_).index(label)`.
- `get_params` reports `estimator=None` while the final step is the untouched default, so
  `clone` rebuilds an equal detector.
- The estimator tags declare a binary classifier. The classifiers declare 2-D input, and
  `EntropyTransformer`'s tags declare both 2-D and 3-D input.

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
- `Judgment.from_text` and `read_judgment` return `None` for a reply that is not JSON and
  carries both `"judgment": true` and `"judgment": false`, instead of reading it as correct.
- A chat choice whose `logprobs.content` is null is read as a sequence without tokens, as one
  whose `logprobs` is null is, instead of raising a `ValidationError`.
- The `adapters` extra requires `langfuse<5`: 5.0 removed the trace calls
  `HallucinationEvaluator` reads with.
