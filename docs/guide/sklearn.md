# scikit-learn compatibility

`EPR` and `WEPR` are scikit-learn classifiers over logprob arrays. Each holds a small pipeline,
`entropy -> classifier`, in `pipeline_`; they are not `Pipeline` subclasses.
`LogProbParser(k=15).transform(responses)` makes the arrays from completion responses, and
`make_detector(k=15)` chains the parser and a `WEPR` into a `DetectorPipeline` over responses: a
`Pipeline` that also has `predict_token_proba`.
`clone`, `get_params`, `cross_val_score` and `GridSearchCV` work as they do for any estimator.

The package requires scikit-learn 1.8 or later (the `pyproject.toml` floor, set by the
published detectors). The estimator checks run in CI on the 1.8 floor and on the release
the lockfile pins, currently 1.9; releases in between are expected to behave the same, but
are not tested.

## What is checked

`EPR`, `WEPR`, a custom detector written as in [Write your own detector](#write-your-own-detector)
and `EntropyTransformer` run scikit-learn's own estimator checks (`parametrize_with_checks`) in
[`tests/test_common.py`](https://github.com/artefactory/artefactual/blob/main/tests/test_common.py),
with no expected failure. `LogProbParser` runs the checks that need no data, called by hand: it
reads responses, which are not an array, and every check that sends data feeds arrays, which the
parser refuses. The classifiers, which have an array entry point, stand for the detectors.

## Known limitations

- **`k` pins the width of an `EPR`.** `EPR(k=15)` refuses input whose last dimension is not 15
  (`trained on 15 candidates per token, got 10: use LogProbParser(k=15)`), including a loaded or
  cloned one. Without `k` an `EPR` accepts any width and, having one feature, scores another width
  without an error: the scores are then wrong. `EPR.from_pretrained` requires `k`. `WEPR` has no `k`:
  its coefficients fix the width and another one is refused.
- **`classes_` is sorted.** `predict_proba` columns follow it, so column 1 is the
  hallucination class only for `0`/`1` labels.
- **Targets are binary.** A multi-class `y` raises `ValueError` at `fit`.
- **Responses are refused.** `EPR` and `WEPR` take logprob arrays; a list of responses raises
  `TypeError` naming `make_pipeline(LogProbParser(k=15), EPR(...))`.

## Inputs

`fit`, `predict`, `predict_proba` and `predict_token_proba` take a logprob array: a
`(n_sequences, n_tokens, width)` array, `NaN`-padded, which `LogProbParser` emits, or a numeric
2-D table, in which each row is a one-token sequence and each column that token's logprob at one
rank. The responses a `LogProbParser` reads are what an OpenAI-compatible client returns with
`logprobs=True` and `top_logprobs` set to `k` ({doc}`scoring`); the accepted shapes are in
{doc}`reference`.

Targets must be binary, and the tags say so: `get_tags(WEPR()).classifier_tags.multi_class` is
`False`.

```python
import numpy as np

from artefactual import EPR, WEPR

rng = np.random.default_rng(0)
logprobs = -rng.exponential(size=(40, 15)).cumsum(axis=1)  # 40 one-token rows, 15 ranks
y = rng.integers(0, 2, size=40)

classifier = WEPR().fit(logprobs, y)
classifier.predict_proba(logprobs)[:3, 1]

epr = EPR(k=15).fit(logprobs, y)
epr.predict_token_proba(logprobs).shape  # (40, 1, 1)
```

From responses, the code below uses the two responses shipped with the repository, repeated to
make a training set big enough to split. They are a stand-in for your own labelled responses.

```python
import json

from artefactual import WEPR, LogProbParser, make_detector

path = "docs/examples/open_ai_responses_top15.json"
RESPONSES = json.load(open(path))["responses"]
responses = RESPONSES * 10
labels = np.array([0, 1] * 10)

X = LogProbParser(k=15).transform(responses)
WEPR().fit(X, labels).predict_proba(X)[:, 1]

detector = make_detector(k=15).fit(responses, labels)  # the same, from responses
detector.predict_proba(RESPONSES)[:, 1]
detector.predict_token_proba(RESPONSES).shape  # (2, max_tokens, 1)

# Published weights, end to end
make_detector(k=15, classifier=WEPR.from_pretrained("artefactory/wepr-phi4")).predict_proba(RESPONSES)[:, 1]
```

## Feature names

`get_feature_names_out` names the pooled features: `epr` for `EPR`;
`wepr_mean_rank0`..`wepr_mean_rank{k-1}` then `wepr_max_rank0`..`wepr_max_rank{k-1}` for
`WEPR`. A classifier fitted on a DataFrame names them after its columns instead,
`wepr_mean_<column>` then `wepr_max_<column>`. `EntropyTransformer(pooling)` takes no `k`: it reads
the width from the data, and a fitted one refuses another width. A custom pooling function is
named after itself, and a `Pooling` subclass by its `feature_names`.

```python
classifier.get_feature_names_out()[:3]
```

## Persistence

`save_pretrained` writes the trained estimator alone to a `.skops` file, which names only
scikit-learn types and so loads under a different install of artefactual.
`from_pretrained` reads one back from a path, a directory holding `model.skops`, or a
Hugging Face repository id ({doc}`reference` has the file layout) and returns a detector backed by
a `FrozenEstimator`: it predicts, also after `clone`, and a `fit` keeps the weights.

```python
saved = classifier.save_pretrained("my_detector.skops")
again = WEPR.from_pretrained(saved)
again.predict_proba(logprobs)[:3, 1]

epr.save_pretrained("my_epr.skops")
EPR.from_pretrained("my_epr.skops", k=15)  # EPR needs the k it was trained at; see the limitations
```

Do not open an untrusted file with `joblib` or `pickle`: they run code from the file.

## Advanced usage

Everything above uses `EPR`, `WEPR` and `make_detector` directly. The recipes below use
`make_pipeline`, for work that needs the parser and the classifier in one estimator.

### End to end on responses

A pipeline around a published detector scores responses in one call; the weights are frozen, so it
predicts and is not tuned.

```python
from sklearn.pipeline import make_pipeline

pretrained = make_pipeline(LogProbParser(k=15), WEPR.from_pretrained("my_detector.skops"))
pretrained.predict_proba(RESPONSES)[:, 1]
```

### Tuning with `GridSearchCV`

Tune a detector you train, never a published one. Step parameters use the `step__parameter`
convention and `make_pipeline` names the steps `logprobparser` and `wepr`. `k` is searchable as
`logprobparser__k`, and the classifier's own parameters as `wepr__estimator__<name>`. Give `WEPR` an
explicit estimator: the default `None` has no `C` to search.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

search = GridSearchCV(
    make_pipeline(LogProbParser(), WEPR(estimator=LogisticRegression())),
    {"logprobparser__k": [5, 10, 15], "wepr__estimator__C": [0.1, 1.0]},
    cv=3,
).fit(responses, labels)
search.best_params_
search.best_estimator_[-1].save_pretrained("tuned.skops")
```

### Write your own detector

A detector is a scikit-learn classifier that picks the features it needs. Keep a fitted `Pipeline`
as `pipeline_`, write `fit` and `predict*` around it, and inherit `TokenProbaMixin` for `predict_token_proba`, which
scores tokens with `token_proba(self.pipeline_.steps, X)`. The contract:

- `token_proba(steps, X)` walks fitted `(name, estimator)` steps: each step runs `transform_tokens`
  when it implements `TokenTransformer` (an `EntropyTransformer` does), else `transform`. The first
  such step defines the tokens, so later steps and the final `predict_proba` see real tokens only.
  The result is the probability of class 1 per token, `NaN` at padded positions.
- A pipeline without a token step needs a last step implementing `TokenClassifier`, which
  scores tokens itself; otherwise `token_proba` raises `TypeError`.
- Set `classes_` and `n_features_in_` in `fit`, never mutate the constructor parameters, and clone
  the estimator instead of fitting it in place; `check_estimator` verifies these.

A pipeline of steps needs no class: `DetectorPipeline` adds `predict_token_proba` to a `Pipeline`
and checks its layout at `fit`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from artefactual import DetectorPipeline, EntropyTransformer, EPRPooling

pipeline = DetectorPipeline(
    [
        ("entropy", EntropyTransformer(EPRPooling())),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ]
).fit(logprobs, y)
pipeline.predict_token_proba(logprobs)  # (n_sequences, max_tokens, 1)
```

A class gives the detector its own parameters and `predict*`:

```python
import dataclasses

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted

from artefactual import EntropyTransformer, EPRPooling, TokenProbaMixin


@dataclasses.dataclass(frozen=True)
class MyPooling(EPRPooling):
    """One feature per sequence; override `__call__` to pool differently."""


class MyDetector(TokenProbaMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = True
        tags.input_tags.three_d_array = False
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.poor_score = True
        return tags

    def fit(self, X, y):
        check_classification_targets(y)  # refuses continuous and unknown targets
        if type_of_target(y) != "binary":  # the error check_estimator expects for a multi-class y
            raise ValueError("Only binary classification is supported.")
        estimator = LogisticRegression() if self.estimator is None else self.estimator
        self.pipeline_ = make_pipeline(EntropyTransformer(MyPooling()), clone(estimator)).fit(X, y)
        self.classes_ = self.pipeline_.classes_
        self.n_features_in_ = self.pipeline_[0].n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.pipeline_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.pipeline_.predict_proba(X)


check_estimator(MyDetector())  # raises on the first failed check
make_pipeline(LogProbParser(k=15), MyDetector(estimator=RandomForestClassifier())).fit(responses, labels)
```

`check_estimator` and `parametrize_with_checks` are the way to know the class behaves as a
scikit-learn estimator; the package's own detectors pass them with no expected failure.

### Extra features with `FeatureUnion`

Concatenate other features next to the entropy ones, then end on any classifier:

```python
from sklearn.pipeline import FeatureUnion

from artefactual import WEPRPooling

features = FeatureUnion([("wepr", EntropyTransformer(WEPRPooling())), ("epr", EntropyTransformer(EPRPooling()))])
make_pipeline(LogProbParser(k=15), features, LogisticRegression()).fit(responses, labels)
```
