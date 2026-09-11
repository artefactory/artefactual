# How it works

Each detector is a three-step scikit-learn pipeline:

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
`(n_sequences, n_tokens, k)` array of log-probabilities, `NaN`-padded.

## Measure

The middle step turns the raw distribution into a small feature vector
summarising how uncertain the model was. The shipped detectors measure this with the
entropy contribution of each candidate, `s_kj = -p_kj * log(p_kj)` for token *j* and rank
*k*, reduced two ways:

| Reduction | Feature vector | Reads |
|---|---|---|
| `epr` | `mean_j( sum_k s_kj )` — 1 feature | The sequence's average top-`k` entropy |
| `wepr` | `mean_j(s_kj)` ‖ `max_j(s_kj)` — 2k features | Per-rank mean and peak, weighted separately |

EPR sums the rank axis, so it is the top-`k` entropy estimate averaged over the sequence —
which is why `k` belongs to the feature's definition, and why a response carrying fewer
ranks is rejected rather than padded: the missing contributions would simply be absent, and
the response would look more confident than it was.

WEPR weights each rank separately, which pays off because `-p*log(p)` peaks at `p = 1/e`: a
mid-ranked candidate carries more signal than either the near-certain top rank or the
negligible tail.

## Classify

A `LogisticRegression` maps the features to a probability. It is fitted on 0/1 labels,
one per generated sequence, where 1 marks a hallucination; the published detectors
are that same estimator with its coefficients already fitted, loaded by
`BaseDetector.from_pretrained`. Because it is a real sklearn classifier, `predict`,
`predict_proba` and `decision_function` all work as expected.

The measurement step is a plain transformer, so any callable can replace the reduction:

```python
import numpy as np

from artefactual.scoring import EntropyTransformer

EntropyTransformer(reduction=lambda s, axis: np.nanmax(s, axis=axis))
```

Fitting one of these for a model with no published weights is {doc}`training`.
