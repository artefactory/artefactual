# Artefactual

**Artefactual** scores how likely an LLM response is to be a hallucination, using only the
log-probabilities the model already returns — no extra generations, no second model, no
access to weights.

Detectors are `scikit-learn` pipelines, so they compose with the tooling you already use.

```python
from artefactual.scoring import wepr

detector = wepr("mistralai/Ministral-8B-Instruct-2410")
detector.predict_proba(response)[:, 1]      # P(hallucination) per sequence
detector.predict_token_proba(response)      # ...and per token
```

## Features

- **Cheap**: one pass over the response you already generated, no second opinion to buy
- **Practical**: calibrations shipped for several model families, or fit your own
- **Flexible**: reads the OpenAI Chat Completions and Responses API formats
- **Detailed**: sequence-level and token-level scores, for highlighting the spans that
  drove a low score

## Detectors

Two ship, differing in how much of the response's confidence they read:

- **`wepr(...)`** — weights each rank of the token distribution separately. The stronger
  detector, and the one to default to.
- **`epr(...)`** — a single pooled feature. Worth it only when there is too little
  labelled data to fit WEPR's larger coefficient vector.

Both are calibrated the same way and on the same data, so choosing `epr` saves no setup
work. Both take the same arguments and return the same thing, so switching between them is
a one-word change. See the
[README](https://github.com/artefactory/artefactual#how-it-works) for what each measures.

## Requirements

Responses must carry `logprobs` with at least `k` entries in `top_logprobs`, where `k` is
the rank count the calibration was fit at (15 for every shipped file). Narrower responses
are rejected rather than scored, since the missing ranks cannot be reconstructed.

```{toctree}
:maxdepth: 2

examples/index
api
presentations/index
```
