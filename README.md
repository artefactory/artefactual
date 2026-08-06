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
generation.

## Limitations

Worth knowing before you adopt it:

- **You need `logprobs` and `top_logprobs`.** Providers that do not expose them cannot be
  scored at all. `top_logprobs` must be at least the calibrated `k` (15), which also rules
  out providers capping it lower.
- **Calibrations are model-specific.** The coefficients map entropy to probability for one
  model's distribution. Four are shipped (see [Supported models](#supported-models));
  scoring a different model needs [your own calibration](#calibrating-on-your-own-data),
  which needs labelled data.
- **The probability is only as good as its calibration.** Rankings transfer more readily
  than absolute values — if you only need to triage the riskiest answers, the raw ordering
  is the robust part.
- **It detects distributional uncertainty, not falsehood.** A model that is confidently
  wrong — a memorised misconception, a poisoned fine-tune — has a peaked distribution and
  scores low. This is a complement to retrieval grounding or a judge, not a replacement.
- **Scoring is per sequence.** There is no cross-response consistency check, which is the
  signal sampling-based detectors buy with their extra generations.

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
    top_logprobs=15,  # required: at least the rank count the weights were calibrated at
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
| `epr` | `mean_j( mean_k s_kj )` — 1 feature | Mean entropy contribution per rank, per token |
| `wepr` | `mean_j(s_kj)` ‖ `max_j(s_kj)` — 2k features | Per-rank mean and peak, weighted separately |

EPR treats every rank alike; the calibrated coefficient is named `mean_entropy` because
the feature is a *mean* over the `k` ranks, not a sum. Since the denominator is `k` rather
than however many ranks a response happens to carry, `k` is part of the feature's
definition rather than a display detail — which is why responses narrower than `k` are
rejected instead of padded.

WEPR learns a weight per rank, which matters because `-p*log(p)` peaks at `p = 1/e` —
mid-ranked candidates are more informative than either the near-certain top rank or the
negligible tail.

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

Both `epr()` and `wepr()` accept any of these names.

### The `k` parameter

`k` is the top-k rank count you intend to score at, and it defaults to **15** — the rank
count every shipped file was calibrated at.

```python
detector = wepr("microsoft/phi-4", k=15)
```

**Your responses must carry at least `k` ranks.** This is an input requirement, not
something the pipeline reconciles for you — generate with `top_logprobs` set to `k` or
higher. The two directions are not symmetric:

- **Wider than `k`** — surplus ranks are dropped. The calibration never saw them, and a
  mean over `k` ranks is defined without them, so scoring is unaffected.
- **Narrower than `k`** — refused. Those ranks are not absent from the distribution, only
  unfetched, so filling them with zeros understates the entropy by exactly `width/k`:

  ```
  ValueError: Response 0 carries 5 rank(s) per token but k=15 was requested. The missing
  ranks are not absent from the distribution, only unfetched, so zero-filling them would
  understate the entropy by a factor of 5/15. Regenerate with top_logprobs=15, or score
  at k=5 with a calibration fit at that rank count.
  ```

  The resulting score is wrong rather than merely rescaled — a narrow response can score
  as *more* confident than a genuinely wider one — so it is rejected instead of adjusted.
  Each response is checked individually, so a wide response cannot mask a narrow one in
  the same batch.

WEPR additionally validates the weights themselves, since its coefficient vector has one
entry per rank:

```
ValueError: Weights cover 15 rank(s) but k=20 was requested. WEPR coefficients are
fixed at the rank count used during calibration; pass k=15, or supply weights
calibrated at k=20.
```

EPR calibrations record no rank count, so for `epr()` the `k` you pass is what governs the
input width.

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

### Calibrating on your own data

Any model that returns `top_logprobs` can be calibrated, not just the four shipped ones.
Ask the same factory for an unfitted detector and fit it on 0/1 labels, where 1 marks a
hallucination:

```python
detector = wepr(k=15, trainable=True).fit(responses, y)
coefficients = detector.named_steps["classifier"].coef_
```

`trainable=True` is explicit on purpose: calling `wepr()` with neither weights nor
`trainable=True` raises rather than quietly returning a detector that would train on your
data and emit probabilities no calibration backs.

To go from a question set to a calibration end to end — generating answers, labelling them
with an LLM judge, fitting and evaluating — see
[`scripts/ecir/README.md`](scripts/ecir/README.md). Both LLM stages run under
`vllm run-batch`; the scripts only prepare its input and read its output.

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

See `docs/examples/langfuse_integration_demo.ipynb`.

## Examples

| Notebook | Shows |
|---|---|
| `docs/examples/epr_usage_demo.ipynb` | Scoring with EPR end to end |
| `docs/examples/wepr_usage_demo.ipynb` | WEPR, including token-level highlighting |
| `docs/examples/langfuse_integration_demo.ipynb` | Scoring observability traces |

Run them without a GPU or any model download:

```bash
uv run jupyter lab docs/examples/epr_usage_demo.ipynb
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
uvx --from shellcheck-py shellcheck scripts/ecir/*.sh
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
