# How it works

A **detector** is `WEPR` or its `EPR` variant, scoring the arrays `LogProbParser(k=15)` makes from
responses; `make_detector(k=15)` chains the two. Each is a scikit-learn estimator that passes the
library's own estimator checks ({doc}`sklearn`):

```{mermaid}
flowchart LR
    R["The answer, and the<br/>probabilities behind it"]
    P["Parse"]
    E["Measure<br/>the uncertainty"]
    C["Classify"]
    O["P(hallucination)"]

    R --> P --> E --> C --> O
```

The three middle boxes are the three sections below.

## Parse

`LogProbParser` normalises OpenAI Chat Completions and Responses API payloads
— as SDK objects or plain dicts, singly or batched — into a dense
`(n_sequences, n_tokens, k)` array of log-probabilities, `NaN`-padded. A numeric 2-D array is
refused, since scoring logprobs you already hold is what `EPR` and
`WEPR` are for: they skip the parser and read each row as a one-token sequence,
reshaped to `(n, 1, width)`.

Two words for that last axis, and they are not the same thing. At every token position the
caller asks the endpoint for the `k` most likely **candidates** with `top_logprobs=k`: a
candidate is a token offered at that position. A candidate's **rank** is its place among
those `k`, counted from 0 as the code counts it: rank 0 is the most likely, rank 14 the
least when `k` is 15. EPR pools all the candidates into one number; WEPR fits one
coefficient per rank, so it can weight some ranks more than others.

## Measure

The middle step turns the raw distribution into a small feature vector
summarising how uncertain the model was. The shipped detectors measure this with the
entropy contribution of each candidate, `s_kj = -p_kj * log(p_kj)` for the candidate at
rank *k* of token *j*, pooled two ways:

| Pooling | Feature vector | Reads |
|---|---|---|
| `epr` (`EPR`) | `mean_j( sum_k s_kj )` — 1 feature | The sequence's average top-`k` entropy |
| `wepr` (`WEPR`) | `mean_j(s_kj)` ‖ `max_j(s_kj)` — 2k features | Per-rank mean and peak, weighted separately |

EPR sums the rank axis, so it is the top-`k` entropy estimate averaged over the sequence.
`k` is part of the feature's definition, and a response carrying fewer candidates is
rejected: the missing contributions would be absent and the response would look more
confident than it was.

WEPR weights each rank separately, which pays off because `-p*log(p)` peaks at `p = 1/e`: a
mid-ranked candidate carries more signal than either the near-certain top rank or the
negligible tail.

## Classify

A `LogisticRegression` maps the features to a probability. It is fitted on 0/1 labels,
one per generated sequence, where 1 marks a hallucination; the published detectors
are that same estimator with its coefficients already fitted, loaded by
`EPR.from_pretrained` or `WEPR.from_pretrained`. Because it is a real sklearn classifier, `predict`,
`predict_proba` and `decision_function` all work as expected, and so do `classes_` and the
feature names: `detector.get_feature_names_out()` names the measured features ({doc}`sklearn`).

The measurement step is a plain transformer, and a pooling is a `Pooling` instance, or any
function `f(x, axis) -> (..., n_features)`: `x` holds the NaN-padded entropy contributions
`s_kj`, its last axis is the rank axis, and `axis` is the token axis to pool. `EPR` and `WEPR`
use `EPRPooling()` and `WEPRPooling()`. Pass a bare function when its name is enough:

```python
import numpy as np

from artefactual import EntropyTransformer


def median(x, axis):
    per_token = np.nansum(x, axis=-1, keepdims=True)
    return np.nanmedian(per_token, axis=axis)


EntropyTransformer(pooling=median)  # one feature, named "median"
```

Subclass `Pooling` to name the features after the rank columns, to declare the width, or to
carry parameters. The class name, lower-cased, is the pooling's `name`. Decorate the subclass
with `@dataclasses.dataclass(frozen=True)` and declare its parameters as fields: equality, hash
and `repr` follow from them, which is what `clone` and the refit check rely on; an undecorated subclass raises `TypeError` when instantiated. A field that
holds an array is declared `dataclasses.field(compare=False)` so it stays out of both:

```python
import dataclasses

from artefactual import Pooling


@dataclasses.dataclass(frozen=True)
class RankMedian(Pooling):
    def __call__(self, X, axis):
        return np.nanmedian(X, axis=axis)

    def feature_names(self, ranks):
        return [f"med_{r}" for r in ranks]


EntropyTransformer(pooling=RankMedian())
```

Without `feature_names`, a pooling is named after itself for one feature and `name_0`,
`name_1`, ... for more. The feature count is read by running the pooling on zeros; override
`width(k)` to declare it instead, and set `constant_width = True` when it does not depend on
`k`. There is no registry: a pooling is always passed as an object, never selected by name.

Fitting one of these for a model with no published weights is {doc}`training`.
