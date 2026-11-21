# Using a detector

A *response* is the result of a call to an OpenAI-compatible API (chat completions, the
Responses API, or a batch output line) made with `logprobs=True` and `top_logprobs`. The
examples below score one, called `response`.

## Choosing a detector

`WEPR` is the more accurate of the two and the one the README uses. Both detectors are
trained the same way and on the same labelled data, so `EPR` saves no preparation work —
only parameters. There is no default detector: you import the class you want.

```python
from artefactual import EPR, WEPR
```

| | `WEPR` | `EPR` |
|---|---|---|
| Features | 2 × `k` | 1 |
| Reads | Each rank separately | Overall confidence per token |
| Output | `predict_proba`, a probability | `predict_proba`, a probability |
| Feature the classifier sees | A mean and a max per rank | The raw entropy rate, pooled over every candidate |
| Ships pre-trained | For the models in the README's table | For the same models |
| Training needed | Only for another model | Only for another model |
| Applicable when | Almost always | Too few labelled examples to fit `2k` coefficients |

Both accept the same arguments and return the same type, so switching between them is a
one-word change. Neither takes a string option: the choice is the class.

## Turning a score into a decision

There is no default threshold, and the package does not pick one. The paper evaluates with
ROC-AUC and PR-AUC, both of which are threshold-free, so no operating point is published;
the right cut depends on the base rate of hallucination in the traffic and on the relative
cost of a false flag. Choose it on a labelled sample of your own responses.

Both detectors end in a logistic regression, so both `predict_proba` outputs lie in
`[0, 1]` and rank responses. How far an output can be read as an absolute probability
depends on how representative the training sample was, not on the detector.

Column order follows `detector.classes_`: with `0`/`1` labels, column 1 is the hallucination
class. With other labels, look the column up; {doc}`sklearn` shows how.

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

from artefactual import WEPR
from artefactual.adapters.langfuse.evaluator import HallucinationEvaluator

evaluator = HallucinationEvaluator("wepr", get_client(), WEPR.from_pretrained("artefactory/wepr-phi4", k=15))
evaluator.score_trace(trace_id)
```

A worked version is in the [example notebooks](../examples/index.md).

## Training a detector for another model

Any model that returns `top_logprobs` can be scored, not only the published ones.
`WEPR(k=15)` and `EPR(k=15)` return an unfitted detector, which is fitted on 0/1 labels where 1
marks a hallucination:

```python
# responses: a list of completion responses, each generated with top_logprobs >= k
# y:         a matching list of 0/1 labels, 1 marking a hallucination
detector = WEPR(k=15).fit(responses, y)
coefficients = detector.named_steps["classifier"].coef_
```

One label per generated sequence, in the same order `predict_proba` returns rows. The
labels must be binary; `classes_` is set from them at `fit`.

A detector is unfitted until `fit`, as every scikit-learn estimator is. Published weights
are never a constructor default — a detector's coefficients are model-specific, so there is
nothing sensible to default to — and `EPR.from_pretrained` / `WEPR.from_pretrained` are
what load them.

Producing those answers and verdicts, the label polarity a judge's reply needs, and the
three notebooks that do it end to end are in {doc}`training`.

## Composing with scikit-learn

Detectors are `sklearn.pipeline.Pipeline` subclasses, so introspection and composition
work:

```python
from sklearn.base import clone

detector = WEPR.from_pretrained("artefactory/wepr-ministral", k=15)
detector.named_steps  # {'parser': ..., 'entropy': ..., 'classifier': ...}
clone(detector)  # get_params / set_params round-trip
```

A detector takes raw responses in `fit` and in the scikit-learn tools built on it, so
`cross_val_score(WEPR(k=15), responses, y)` and `GridSearchCV` work without parsing first.
{doc}`sklearn` has the model-selection recipes, the feature names and the persistence
rules.

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

A response carrying fewer candidates per token than the detector was trained at:

```
ValueError: Response 0 carries 5 candidate(s) per token but k=15 was requested. The missing
candidates are not absent from the distribution, only unfetched, so zero-filling them would
drop their entropy contributions and score the response as more confident than it was.
Regenerate with top_logprobs=15, or score at k=5 with a detector trained at that width.
```

Why the second is refused rather than padded is in
[the reference](reference.md#the-k-parameter).
