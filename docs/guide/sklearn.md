# scikit-learn compatibility

`EPRClassifier` and `WEPRClassifier` are scikit-learn classifiers: subclasses of
`sklearn.pipeline.Pipeline` with an `entropy -> classifier` chain over logprob arrays. `EPR`
and `WEPR` are the same estimators with a `LogProbParser` in front, so they take completion
responses: a `parser -> entropy -> classifier` chain. `clone`, `get_params`,
`GridSearchCV`, `cross_val_score` and `Pipeline` slicing all work as they do for any
estimator.

The package requires scikit-learn 1.8 or later (the `pyproject.toml` floor, set by the
published detectors). The estimator checks run in CI on the 1.8 floor and on the release
the lockfile pins, currently 1.9; releases in between are expected to behave the same, but
are not tested.

## What is checked

The classifiers run scikit-learn's own estimator checks (`parametrize_with_checks`) in
[`tests/test_common.py`](https://github.com/artefactory/artefactual/blob/main/tests/test_common.py).
All pass but one, which is declared and enforced as a strict expected failure:

- `check_estimators_overwrite_params`. A `Pipeline` fits its steps in place, which the
  check reads as the constructor parameters being overwritten. scikit-learn lists the
  same check as an expected failure for `Pipeline` itself, so the classifiers inherit the
  limitation rather than add one. If a scikit-learn release fixes it, the strict marker
  turns the suite red and the entry is removed.

`LogProbParser`, `EntropyTransformer`, `EPR` and `WEPR` are not in
`parametrize_with_checks`. The parser and the detectors read responses, which are not an
array, and the entropy step consumes a cube of three dimensions, so scikit-learn's checks,
which feed 2-D arrays, do not apply to them. The classifiers, which have a 2-D entry point,
are what is checked.

## Two kinds of input

The two pairs of estimators differ in what `fit`, `predict`, `predict_proba` and
`predict_token_proba` take:

- **Responses**, for `EPR` and `WEPR`: a list of what an OpenAI-compatible client returns
  with `logprobs=True` and `top_logprobs` of at least `k`. This is what you score in
  practice. The accepted shapes are in {doc}`reference`. A 1-D array or Series of responses
  is read as responses. A numeric table raises `TypeError`, which names the classifiers.
- **Logprob arrays**, for `EPRClassifier` and `WEPRClassifier`: a
  `(n_sequences, n_tokens, width)` array, `NaN`-padded, or a numeric 2-D table, in which
  each row is a one-token sequence and each column that token's logprob at one rank. Ranks
  beyond `k` are dropped, the least likely candidates first, so one wide array serves a
  search over `k`. A narrower input is used as it is. This is the form the estimator checks
  use, and it lets the estimator sit behind tools that only know arrays. The array path
  expects log-probabilities (at most 0) but does not check the sign, because
  `check_estimator` feeds arbitrary floats; responses are validated, tables are not. A 2-D
  object table is read as numbers.

```python
import json

import numpy as np

from artefactual import WEPR, WEPRClassifier

rng = np.random.default_rng(0)
logprobs = -rng.exponential(size=(40, 15)).cumsum(axis=1)   # 40 one-token rows, 15 ranks
y = rng.integers(0, 2, size=40)

classifier = WEPRClassifier().fit(logprobs, y)
classifier.predict_proba(logprobs)[:3, 1]
```

Targets must be binary. A multi-class `y` raises `ValueError` at `fit`, and the tags say
so: `get_tags(WEPRClassifier()).classifier_tags.multi_class` is `False`, and the input is
declared as a 2-D array.

The code below uses the two responses shipped with the repository, repeated to make a
training set big enough to split. They are a stand-in for your own labelled responses.

```python
path = "docs/examples/open_ai_responses_top15.json"
RESPONSES = json.load(open(path))["responses"]
responses = RESPONSES * 10
labels = np.array([0, 1] * 10)

detector = WEPR().fit(responses, labels)
detector.predict_proba(RESPONSES)[:, 1]
```

## Which column is the hallucination class

Columns of `predict_proba` follow `classes_`, sorted. With `0`/`1` labels, column 1 is the
hallucination class. With any other labels, look the column up instead of assuming it:

```python
column = list(detector.classes_).index(1)
detector.predict_proba(RESPONSES)[:, column]
```

See {doc}`scoring` for how the score is read.

## Model selection

Because a detector is an estimator over responses, the scikit-learn tools take the
responses directly. Step parameters use the `step__parameter` convention, and the steps
are named `parser`, `entropy` and `classifier`. A classifier has the last two:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

cross_val_score(WEPR(), responses, labels, cv=3)

search = GridSearchCV(
    WEPR(),
    {"k": [5, 10, 15], "classifier__C": [0.1, 1.0, 10.0]},
    cv=3,
).fit(responses, labels)
search.best_params_
```

`k` is the parameter to sweep: setting it reaches the parser and the entropy step. Setting
`parser__k` alone leaves the entropy step at its old `k`, so the feature names no longer
match the features. A `k` below the width of the responses (the candidates per token they carry) drops the
surplus candidates. Above it, the parser refuses the responses, since the missing
candidates were never fetched.

Setting both the estimator and one of its parameters in one grid fails exactly as it does
for scikit-learn's `Pipeline`: `set_params(estimator=None, estimator__C=...)` sets the
parameter before the step exists. Use an estimator instance in grids that set both, for
example `{"estimator": [LogisticRegression(C=1.0)], ...}`.

Sample weights reach the logistic regression through the pipeline's own prefix:

```python
weights = np.linspace(1.0, 2.0, len(labels))
WEPR().fit(responses, labels, classifier__sample_weight=weights)

GridSearchCV(WEPR(), {"classifier__C": [0.1, 1.0]}, cv=3).fit(
    responses, labels, classifier__sample_weight=weights
)
```

## Feature names and DataFrame output

Slices of an estimator are transformers, so scikit-learn's `set_output` applies to them.
The columns are named from `k`: `epr` for `EPR` and `EPRClassifier`;
`wepr_mean_rank0`..`wepr_mean_rank{k-1}` then `wepr_max_rank0`..`wepr_max_rank{k-1}` for
`WEPR` and `WEPRClassifier`. A classifier fitted on a DataFrame names them after its
columns instead, `wepr_mean_<column>` then `wepr_max_<column>`:

```python
front = detector[:-1].set_output(transform="pandas")  # or "polars"
front.get_feature_names_out()
front.transform(RESPONSES)  # a DataFrame with those columns
```

The parser yields a three-dimensional array that has no table form, so it is never
converted; only the entropy step's `transform` is. An estimator with a custom reduction
callable has no feature names.

## Persistence

`save_estimator` writes the fitted classifier alone to a `.skops` file, which names only
scikit-learn types and so loads under a different install of artefactual.
`from_pretrained` reads one back from a path, a directory holding `model.skops`, or a
Hugging Face repository id ({doc}`reference` has the file layout):

```python
saved = detector.save_estimator("my_detector.skops")
again = WEPR.from_pretrained(saved)
```

Dumping the whole detector with `skops.io.dump` also works, but loading it names the
artefactual types it holds, which you pass as `trusted=`. Ask skops for the exact list:

```python
from skops.io import dump, get_untrusted_types, load

dump(detector, "whole.skops")
trusted = get_untrusted_types(file="whole.skops")
whole = load("whole.skops", trusted=trusted)
```

Do not open an untrusted file with `joblib` or `pickle`: they run code from the file. The
`trusted=` list is for files you produced yourself.
