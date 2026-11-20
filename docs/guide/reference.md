# Reference

Detail behind the width a detector is pinned to, the layout of a weights file,
and the inputs the parser accepts.

## The `k` parameter

`k` is how many **candidates** per token are read — the tokens the model considered at that
position. A candidate's **rank** is its place in that list, counted from 0 for the most
likely, so `k` is also the width of the rank axis. It defaults to **15**, the width every
shipped file was trained at.

```python
detector = WEPR.from_pretrained("artefactory/wepr-phi4", k=15)
```

**Responses must carry at least `k` candidates per token.** This is an input requirement rather than
something the pipeline reconciles: generation must set `top_logprobs` to `k` or higher. The
two directions are not symmetric:

- **Wider than `k`** — surplus candidates are dropped. The detector never saw them, and a
  mean over `k` ranks is defined without them, so scoring is unaffected.
- **Narrower than `k`** — refused. Those candidates are not absent from the distribution, only
  unfetched, so filling them with zeros drops their contributions from the sum:

  ```
  ValueError: Response 0 carries 5 candidate(s) per token but k=15 was requested. The
  missing candidates are not absent from the distribution, only unfetched, so zero-filling
  them would drop their entropy contributions and score the response as more confident than
  it was. Regenerate with top_logprobs=15, or score at k=5 with a detector trained at that
  width.
  ```

  The resulting score is wrong rather than merely rescaled — a narrow response can score
  as *more* confident than a genuinely wider one — so it is rejected instead of adjusted.
  Each response is checked individually, so a wide response cannot mask a narrow one in
  the same batch.

`from_pretrained` also checks the weights against `k`, since a WEPR coefficient vector has
two entries per rank:

```
ValueError: The WEPR detector at 'weights.skops' takes 30 feature(s), but k=20 needs 40.
Its coefficients are fixed at the width they were trained at; pass k=15, or use a detector
trained at k=20.
```

An EPR detector pools every candidate into one feature, so its file records no width and `k`
only sizes the parser. A classifier given an array wider than `k` keeps the first `k` ranks,
and a narrower one is used as it is; {doc}`sklearn` has the details.

## Detector files

A locally trained detector is loaded by passing a path instead of a name:

```python
detector = WEPR.from_pretrained("/path/to/my_detector.skops")
```

`save_estimator` and `from_pretrained` are the entry points: the first writes the fitted
estimator alone, the second reads it back. A detector file is the skops dump of a fitted
`LogisticRegression`, named `model.skops` when it sits in a directory; the coefficients are
in `coef_` and `intercept_`, and nothing else is stored. It names only scikit-learn types, so
it loads under any install of artefactual.

Dumping the whole detector with `skops.io.dump` also works, but loading it needs `trusted=`
listing the artefactual types it holds. Ask skops for that list with
`skops.io.get_untrusted_types(file=path)`. Do not open an untrusted file with `joblib` or
`pickle`: they run code from the file ({doc}`sklearn` covers persistence).

## Input formats

Three inputs are accepted, as SDK objects or plain mappings: a chat completion, a
Responses API payload, and one line of an OpenAI Batch output file -- which is also what
`vllm run-batch` writes, so a batch file is scored as it is read. Numeric arrays go to
`EPRClassifier` and `WEPRClassifier`, described in {doc}`sklearn`. A batch line whose
request failed is *refused*, not skipped: this step emits one row per line, and dropping
one would shift every later response against its label. Filter the failed lines out first,
where their `custom_id` can still be reported.

`artefactual.preprocessing` owns that reading:

```python
from artefactual.preprocessing import Judgment, index_by_custom_id, read_batch, read_judgment, read_message

rows = read_batch("responses.jsonl")     # validates each line; refuses a repeated custom_id
for row in rows:
    if row.failure:                      # why this line carries no completion, or None
        print(f"dropped {row.custom_id}: {row.failure}")

responses = index_by_custom_id(rows)     # the lines that carry a completion, keyed to join on
read_message(row.completion)             # what the model said, or None
read_judgment(row.completion)            # a judge's True / False verdict, or None
Judgment.from_completion(row.completion) # the verdict and the explanation, or None
Judgment.from_text(reply)                # the same, from the reply text
verdicts = Judgment.from_batch(rows)     # one per row, in row order, None where unreadable
```

`Judgment` is also the schema the judge prompt asks for, `{"judgment": true|false,
"explanation": "..."}`, so it can be passed as a structured-output `response_format`.
`judgment=True` means the judge called the answer *correct*, the opposite of the class a
detector predicts. The readers accept a reply that wraps the object in a Markdown fence, or
that is a bare `true` / `false`, and return `None` for anything else rather than a guess;
a stringified `"false"` is not a verdict. `Judgment.from_batch` keeps its result aligned
with `rows`, so `zip(rows, verdicts)` joins on `custom_id`.

`failure` names each way a line can carry nothing -- a non-HTTP error, a status outside
2xx, a missing envelope or body -- and `completion` is `None` for every one of them, so a
caller counts failures in one place instead of branching on which kind it got.

The extraction is a method of the payload models. `ChatCompletion` and `ResponsesPayload`
(in `artefactual.preprocessing.response_models`) share it through `LogprobsMixin`:

```python
from artefactual.preprocessing.response_models import LogprobsMixin

payload = LogprobsMixin.parse(row.model_dump())  # a completion, a Responses payload or a batch line
# ChatCompletion.parse and ResponsesPayload.parse accept their own format only
payload.top_logprobs()      # per sequence: {token position: candidate logprobs, highest first}
payload.sampled_logprobs()  # per sequence: the sampled tokens' logprobs, as a 1-D array
```

`parse_top_logprobs`, `parse_sampled_token_logprobs` and `LogProbParser` are these calls with
the batching and padding around them, and raise the same errors. `parse` refuses a failed batch
line with a `ValueError` naming its `custom_id`: skipping it would shift every later response
against its label.

A minimal Responses API payload looks like:

```python
response = {
    "object": "response",
    "output": [
        {
            "content": [
                {
                    "logprobs": [
                        {"top_logprobs": [{"logprob": -0.1}, {"logprob": -2.3}]},
                        {"top_logprobs": [{"logprob": -0.05}, {"logprob": -3.1}]},
                    ]
                }
            ]
        }
    ],
}
```

Malformed input fails loudly. A payload that is not a completion in any of these formats
raises `TypeError`. A completion whose `top_logprobs` are absent or empty, which is what a
provider returns when logprobs were not requested, raises `ValueError`, as does a positive
or non-finite log-probability, naming the offending sample, token and rank.

A 1-D array or Series of responses is read as responses; a numeric 2-D array raises
`TypeError`, which names the classifiers.

The generated signatures for every public object are in the {doc}`../api`.
