# Using a detector

A *response* is the result of a call to an OpenAI-compatible API (chat completions, the
Responses API, or a batch output line) made with `logprobs=True` and `top_logprobs`. The
examples below score one, called `response`.

## The k rule

`k` is the number of top logprobs the detector was trained with. Request that many
`top_logprobs` when generating, and score with the same `k`. A response with fewer
candidates per token is refused; why is in {doc}`reference`. The published detectors were
all trained with `k = 15`.

## Choosing a detector

`WEPR` is the more accurate of the two and the one the README uses. Both are
trained the same way and on the same labelled data, so `EPR` saves no preparation work —
only parameters. You import the class you want and give it the arrays a `LogProbParser`
makes from responses; `make_detector(k=15)` is the ready-made `LogProbParser` then `WEPR` pipeline.

```python
from artefactual import EPR, WEPR, LogProbParser, make_detector
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

## Scoring every token

```python
logprobs = LogProbParser(k=15).transform(response)
detector = WEPR.from_pretrained("artefactory/wepr-phi4")
token_scores = detector.predict_token_proba(logprobs)  # (n_sequences, max_tokens, 1)
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
logprobs = LogProbParser(k=15).transform([response_a, response_b])
scores = detector.predict_proba(logprobs)  # (n_sequences, 2)
```

Each response is validated on its own, so one malformed or too-narrow response is reported
by position rather than quietly changing its neighbours' scores.

## Scoring Langfuse traces

The adapter fetches a trace, scores its output and writes the probability back as a trace
score. Re-runs overwrite rather than duplicate, so the operation is safe to schedule:

```python
from langfuse import get_client

from artefactual import make_detector
from artefactual.adapters.langfuse.evaluator import HallucinationEvaluator

K = 15  # the k the published detectors were trained with

detector = make_detector(k=K)  # a Pipeline over responses; a Pipeline is what the evaluator takes
# fit it on labelled responses first, or pass make_pipeline(LogProbParser(k=K), WEPR.from_pretrained(...))
evaluator = HallucinationEvaluator("wepr", get_client(), detector)
evaluator.score_trace(trace_id)
```

A worked version is in the [example notebooks](../examples/index.md).

## Training a detector for another model

See {doc}`training`.

## Composing with scikit-learn

`EPR` and `WEPR` are scikit-learn classifiers, and `make_detector(k=15)` is a plain
`sklearn.pipeline.Pipeline`, so introspection and composition work:

```python
from sklearn.base import clone

detector = make_detector(k=15)
detector.named_steps  # {'logprobparser': ..., 'wepr': ...}
clone(detector)  # get_params / set_params round-trip
```

{doc}`sklearn` has the advanced recipes: an end-to-end pipeline over responses, tuning `k` and the
classifier with `GridSearchCV`, a custom detector, and the persistence rules.

## When a response is refused

A response that is not scorable is refused by name, so running it through the detector is
the check. For example, one with fewer candidates per token than the detector's `k`:

```
ValueError: Response 0 carries 5 candidate(s) per token but k=15 was requested. The missing
candidates are not absent from the distribution, only unfetched, so zero-filling them would
drop their entropy contributions and score the response as more confident than it was.
Regenerate with top_logprobs=15, or score at k=5 with a detector trained at that width.
```

Why it is refused rather than padded is in
[the reference](reference.md#the-k-parameter).
