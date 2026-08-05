# Artefactual

**Hallucination detection for black-box LLMs, as scikit-learn estimators.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![Paper](https://img.shields.io/badge/arXiv-2509.04492-b31b1b.svg)](https://arxiv.org/abs/2509.04492)

Artefactual scores how likely an LLM response is to be a hallucination, using only the
token log-probabilities the model already returns. No second model, no reference answer,
no access to weights — just the `logprobs` field of a completion response.

Detectors are `sklearn.pipeline.Pipeline` subclasses, so they drop straight into the
tooling you already use.

```python
from artefactual.scoring import epr

detector = epr("mistralai/Ministral-8B-Instruct-2410")
detector.predict_proba(response)[:, 1]      # P(hallucination) per sequence
detector.predict_token_proba(response)      # ...and per token
```

---

## Why

Sampling-based detectors (SelfCheckGPT and friends) need *n* extra generations per answer.
LLM-as-judge needs a second, usually larger, model. Both cost latency and money at
inference time.

Artefactual reads a signal that is already in the response. When a model is about to
hallucinate, its next-token distribution flattens — the entropy of the top-k candidates
rises. Aggregating those per-token entropy contributions gives a score, and a small
logistic regression, calibrated once per model, turns that score into a probability.

The result is a detector that costs one dot product per token and requires exactly one
generation. The trade-off: you need `logprobs` and `top_logprobs` in the response, which
rules out providers that do not expose them.

## Installation

```bash
pip install artefactual
```

or, for development:

```bash
git clone https://github.com/artefactory/artefactual
cd artefactual
uv sync
```

The core install depends only on `numpy`, `scikit-learn`, `pydantic` and `beartype`.
There is no GPU, torch, or inference-server dependency.

| Extra | Installs | For |
|---|---|---|
| `.[adapters]` | `langfuse`, `openai` | Running the integration examples |
| `.[docs]` | `sphinx` and theme | Building the documentation |

## Quick start

Request log-probabilities when you generate, then score the response object directly —
the pipeline parses it for you.

```python
from openai import OpenAI

from artefactual.scoring import epr

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Who wrote the Rust book?"}],
    logprobs=True,
    top_logprobs=15,  # match the rank count the weights were calibrated at
)

detector = epr("mistralai/Ministral-8B-Instruct-2410")
scores = detector.predict_proba(response)

print(f"P(hallucination) = {scores[0, 1]:.2f}")
```

`predict_proba` returns an `(n_sequences, 2)` array following the scikit-learn convention:
column 0 is P(supported), column 1 is P(hallucinated). One row per generated sequence, so
a request with `n=4` returns four rows.

### Token-level scores

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

This is what you want for highlighting the specific spans that drove a low score.

## How it works

Each detector is a three-step pipeline:

```
LogProbParser  ->  EntropyTransformer  ->  PretrainedLogisticRegression
  response           entropy features         calibrated probability
```

**1. Parse.** `LogProbParser` normalises OpenAI Chat Completions and Responses API payloads
— as SDK objects or plain dicts, singly or batched — into a dense
`(n_sequences, n_tokens, k)` array of log-probabilities, `NaN`-padded.

**2. Entropy.** For each token *j* and rank *k*, the entropy contribution is
`s_kj = -p_kj * log(p_kj)`. `EntropyTransformer` reduces those contributions to features.
Two reductions ship:

| Reduction | Feature vector | Intuition |
|---|---|---|
| `epr` | `mean_j( sum_k s_kj )` — 1 feature | Mean total entropy per token |
| `wepr` | `mean_j(s_kj)` ‖ `max_j(s_kj)` — 2k features | Per-rank mean and peak, weighted separately |

EPR treats every rank alike. WEPR learns a weight per rank, which matters because
`-p*log(p)` peaks at `p = 1/e` — mid-ranked candidates are more informative than either
the near-certain top rank or the negligible tail.

**3. Calibrate.** A `LogisticRegression` pre-loaded with per-model coefficients maps the
features to a probability. Because it is a real sklearn classifier, `predict`,
`predict_proba` and `decision_function` all work as expected.

You can pass any callable as a custom reduction:

```python
import numpy as np

from artefactual.scoring import EntropyTransformer

EntropyTransformer(reduction=lambda s, axis: np.nanmax(s, axis=axis))
```

## Composing with scikit-learn

Detectors are ordinary estimators, so introspection and composition work:

```python
from sklearn.base import clone

detector = wepr("mistralai/Ministral-8B-Instruct-2410")
detector.named_steps  # {'parser': ..., 'entropy': ..., 'classifier': ...}
clone(detector)  # get_params / set_params round-trip
```

Both transformers are stateless — `fit` learns nothing — so you can call `transform`
without fitting. Note that `LogProbParser` sets `no_validation=True` in order to accept
response objects rather than arrays, so cross-validation must start from data that has
already been parsed.

## Supported models

Calibrations ship in `src/artefactual/data` and are addressed by model name:

| Model |
|---|
| `mistralai/Ministral-8B-Instruct-2410` |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` |
| `tiiuae/Falcon3-10B-Instruct` |
| `microsoft/phi-4` |

Both `epr()` and `wepr()` accept any of these names. All shipped weights are calibrated at
**k = 15**, so request `top_logprobs=15`.

To use your own calibration, pass a path instead of a name:

```python
detector = wepr("/path/to/my_weights.json")
```

Weight files are JSON. EPR calibrations carry a single coefficient:

```json
{"intercept": -2.91, "coefficients": {"mean_entropy": 58.17}}
```

WEPR weights carry a `mean_rank_i` and `max_rank_i` pair for each rank `1..k`:

```json
{"intercept": -3.02, "coefficients": {"mean_rank_1": 3.81, "max_rank_1": -0.44}}
```

To calibrate on your own data, fit a `LogisticRegression` on the output of the
`parser -> entropy` steps against 0/1 hallucination labels, then write out its
`intercept_` and `coef_` in that shape.

## Input formats

Both OpenAI wire formats are accepted, as SDK objects or plain mappings, one at a time or
as a list:

```python
detector.predict_proba(response)  # a single response
detector.predict_proba([resp_a, resp_b])  # a batch, concatenated in order
```

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

Malformed input fails loudly: a payload carrying no logprobs raises `TypeError`, and a
positive or non-finite log-probability raises `ValueError` naming the offending sample,
token and rank.

## Integrations

Score Langfuse traces in place:

```python
from artefactual.adapters.langfuse.evaluator import HallucinationEvaluator

evaluator = HallucinationEvaluator("epr", langfuse_client, epr("microsoft/phi-4"))
evaluator.score_trace(trace_id)
```

See `examples/langfuse_integration_demo.ipynb`.

## Examples

| Notebook | Shows |
|---|---|
| `examples/epr_usage_demo.ipynb` | Scoring with EPR end to end |
| `examples/wepr_usage_demo.ipynb` | WEPR, including token-level highlighting |
| `examples/langfuse_integration_demo.ipynb` | Scoring observability traces |

Run them without a GPU or any model download:

```bash
uv run jupyter lab examples/epr_usage_demo.ipynb
```

## API reference

| Object | Purpose |
|---|---|
| `scoring.epr(path)` | Build an EPR detector pipeline |
| `scoring.wepr(path)` | Build a WEPR detector pipeline |
| `scoring.BaseDetector` | The `Pipeline` subclass adding `predict_token_proba` |
| `scoring.EntropyTransformer` | Contributions → features; `reduction` is `"epr"`, `"wepr"` or a callable |
| `scoring.PretrainedLogisticRegression` | `LogisticRegression` loaded from a weights file |
| `preprocessing.LogProbParser` | Response objects → dense NaN-padded array |
| `preprocessing.parse_top_logprobs` | Functional form of the parser |
| `utils.io.load_weights` / `load_calibration` | Registry and file loading |

## Development

```bash
uv sync
uv run pytest tests        # test suite (needs the locked project env)
uv run pytest tests --cov  # with coverage

uvx ruff check src tests   # lint — a standalone tool, no project env needed
uvx ruff format src tests
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If `artefactual` is useful in your research, please cite our paper, accepted for
publication at ECIR 2026:

```bibtex
@misc{moslonka2025learnedhallucinationdetectionblackbox,
      title={Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate},
      author={Charles Moslonka and Hicham Randrianarivo and Arthur Garnier and Emmanuel Malherbe},
      year={2025},
      eprint={2509.04492},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.04492},
}
```

## License

MIT — no limitation of usage, including for commercial applications.
