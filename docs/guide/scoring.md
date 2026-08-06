# Using a detector

The examples below assume a `response` generated with `logprobs=True` and
`top_logprobs=15`, as in the README's quick start.

## Choosing a detector

`wepr` is the default. Both detectors are trained the same way and on the same labelled
data, so `epr` saves no preparation work — only parameters. The training cost is identical
and `wepr` is the more accurate of the two.

| | `wepr(...)` — default | `epr(...)` |
|---|---|---|
| Features | 2 × `k` | 1 |
| Reads | Each rank separately | Overall confidence per token |
| Output | Sigmoid, scaled to `[0, 1]` | Unscaled entropy rate |
| Ships pre-trained | For the four shipped models | For the four shipped models |
| Training needed | Only for another model | Only for another model |
| Applicable when | Almost always | Too few labelled examples to fit `2k` coefficients |

Both accept the same arguments and return the same type, so switching between them is a
one-word change.

## Turning a score into a decision

There is no default threshold, and the package does not pick one. The paper evaluates with
ROC-AUC and PR-AUC, both of which are threshold-free, so no operating point is published;
the right cut depends on the base rate of hallucination in the traffic and on the relative
cost of a false flag. Choose it on a labelled sample of your own responses.

`wepr` is the more usable of the two here: its output is a sigmoid, so it is already scaled
to `[0, 1]` and can be shown to a user as a confidence. `epr` returns an unscaled entropy
rate, which orders responses but carries no absolute meaning.

## Scoring every token

```python
token_scores = detector.predict_token_proba(response)  # (n_sequences, max_tokens, 1)
```

Sequences shorter than the longest in the batch are padded with `NaN`, so mask before
aggregating:

```python
import numpy as np

first = token_scores[0, :, 0]
print(first[~np.isnan(first)])
```

Token-level scores identify the specific spans that drove a low score.

## Scoring a batch

A list scores many responses in one call. They are parsed, padded to a common length and
scored together, one output row per generated sequence in input order:

```python
scores = detector.predict_proba([response_a, response_b])  # (n_sequences, 2)
```

Each response is validated on its own, so one malformed or too-narrow response is reported
by position rather than quietly changing its neighbours' scores.

## Scoring Langfuse traces

The adapter fetches a trace, scores its output and writes the probability back as a trace
score. Re-runs overwrite rather than duplicate, so the operation is safe to schedule:

```python
from langfuse import get_client

from artefactual.adapters.langfuse.evaluator import HallucinationEvaluator

evaluator = HallucinationEvaluator("wepr", get_client(), wepr("chicham/artefactual-wepr-phi4"))
evaluator.score_trace(trace_id)
```

A worked version is in the [example notebooks](../examples/index.md).

## Training a detector for another model

Any model that returns `top_logprobs` can be scored, not only the four shipped ones. The
same factory returns an unfitted detector, which is fitted on 0/1 labels where 1 marks a
hallucination:

```python
# responses: a list of completion responses, each generated with top_logprobs >= k
# y:         a matching list of 0/1 labels, 1 marking a hallucination
detector = wepr(k=15, trainable=True).fit(responses, y)
coefficients = detector.named_steps["classifier"].coef_
```

One label per generated sequence, in the same order `predict_proba` returns rows.

`trainable=True` is explicit by design: calling `wepr()` with neither weights nor
`trainable=True` raises, rather than returning a detector that would fit on the supplied
data and emit probabilities no trained weights support.

Reproducing the paper's end-to-end pipeline — generating answers, labelling them with an
LLM judge, fitting and evaluating — is documented in the
[`scripts/ecir`](https://github.com/artefactory/artefactual/tree/main/scripts/ecir)
subdirectory.

## Composing with scikit-learn

Detectors are `sklearn.pipeline.Pipeline` subclasses, so introspection and composition
work:

```python
from sklearn.base import clone

detector = wepr("chicham/artefactual-wepr-ministral")
detector.named_steps  # {'parser': ..., 'entropy': ..., 'classifier': ...}
clone(detector)  # get_params / set_params round-trip
```

Both transformers are stateless — `fit` learns nothing — so `transform` may be called
without fitting. `LogProbParser` sets `no_validation=True` in order to accept response
objects rather than arrays, so cross-validation must start from already-parsed data.

## When a response is refused

Neither requirement needs auditing in advance: a response that fails one is refused by
name, so running it through the detector is the check.

A response whose provider did not return log-probabilities:

```
ValueError: None of the 1 response(s) carry any log-probabilities. The payload is a
recognised completion format but its top_logprobs are absent or empty, which is what a
provider returns when logprobs were not requested or are unsupported. Regenerate with
logprobs=True and top_logprobs=15.
```

A response carrying fewer ranks than the detector was trained at:

```
ValueError: Response 0 carries 5 rank(s) per token but k=15 was requested. The missing
ranks are not absent from the distribution, only unfetched, so zero-filling them would
drop their entropy contributions and score the response as more confident than it was.
Regenerate with top_logprobs=15, or score at k=5 with a detector trained at that rank
count.
```

Why the second is refused rather than padded is in
[the reference](reference.md#the-k-parameter).
