# Artefactual

**Hallucination detection for black-box LLMs, as scikit-learn estimators.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![Paper](https://img.shields.io/badge/arXiv-2509.04492-b31b1b.svg)](https://arxiv.org/abs/2509.04492)

Artefactual scores how likely an LLM response is to be a hallucination. It reads a
signal the model already emits, so detection costs no extra generations, no second model,
no reference answer and no access to weights — just the `logprobs` field of a completion
response you were going to make anyway.

Detectors are `sklearn.pipeline.Pipeline` subclasses, so they drop straight into the
tooling you already use.

```python
from artefactual.scoring import wepr

detector = wepr("mistralai/Ministral-8B-Instruct-2410")
detector.predict_proba(response)[:, 1]      # P(hallucination) per sequence
detector.predict_token_proba(response)      # ...and per token
```

---

## Features

| | |
|---|---|
| **Score a response** | `predict_proba(response)` → P(hallucination), from the response you already generated |
| **Locate the problem** | `predict_token_proba(response)` → a score per token, for highlighting the spans that drove it |
| **Score in batches** | Pass a list; sequences are parsed, padded and scored together |
| **Bring your own model** | Pre-fit calibrations for four model families, or fit one on your own labelled data |
| **Drop into scikit-learn** | Detectors are `Pipeline` subclasses: `clone`, `get_params`, `GridSearchCV` all work |
| **Read either OpenAI format** | Chat Completions and Responses, as SDK objects or plain dicts |
| **Watch production traffic** | A Langfuse adapter scores traces in place and writes the score back |

No extra generations, no second model, no GPU, and no dependency heavier than
`scikit-learn`.

## Why

Detecting hallucinations usually means paying for a second opinion. Sampling-based
detectors (SelfCheckGPT and friends) need *n* extra generations per answer. LLM-as-judge
needs a second, usually larger, model. Both add latency and cost to every request you want
to check.

Artefactual takes the measurement from the response you already have. A model's own
confidence is informative about whether it is making something up, and that confidence is
already in the `logprobs` — a calibration per model turns it into a probability.

| | Extra generations | Extra model | Cost per check |
|---|---|---|---|
| Sampling-based | *n* | — | *n* × generation |
| LLM-as-judge | — | yes | one judge call |
| **Artefactual** | **none** | **none** | **a dot product per token** |

Scoring is fast enough to run on every response rather than a sample of them, which is
what makes it usable as a filter rather than an audit.

## Results

ROC-AUC on TriviaQA hallucination detection at `k = 15`, from Table 1 of the
[paper](https://arxiv.org/abs/2509.04492):

| Model | SelfCheckGPT | EPR | HalluDetect | WEPR |
|---|---|---|---|---|
| `Mistral-Small-3.1-24B` | 79.0 | 74.6 | 78.7 | **82.0** |
| `Falcon-3-10B` | 70.1 | 75.4 | 79.0 | **84.1** |
| `Phi-4` (14.7B) | 71.4 | 78.2 | 83.8 | **85.4** |
| `Ministral-8B-2410` | 81.1 | 81.4 | **86.1** | 85.8 |

SelfCheckGPT buys its numbers with 10 extra generations per answer; EPR and WEPR use none.
The paper measures roughly 80 ± 20 µs per score, against at least 10 s for SelfCheckGPT.

Scores drop 10-20 points on WebQuestions, a dataset the detectors were not trained on, so
train on data that resembles your traffic — see
[`scripts/ecir/README.md`](scripts/ecir/README.md) for the full tables and the procedure.

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

from artefactual.scoring import wepr

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Who wrote the Rust book?"}],
    logprobs=True,
    top_logprobs=15,  # required: at least the rank count the weights were calibrated at
)

detector = wepr("mistralai/Ministral-8B-Instruct-2410")
scores = detector.predict_proba(response)

print(f"P(hallucination) = {scores[0, 1]:.2f}")
```

`predict_proba` returns an `(n_sequences, 2)` array following the scikit-learn convention:
column 0 is P(supported), column 1 is P(hallucinated). One row per generated sequence, so
a request with `n=4` returns four rows.

## Usage

Everything below assumes a `response` generated with `logprobs=True` and
`top_logprobs=15`, as in the quick start.

### Choosing a detector

**Use `wepr` unless you have a reason not to.** Both detectors are calibrated the same
way, on the same labelled data, so picking `epr` saves no setup work — only parameters.
Given that the training cost is identical, take the stronger detector.

| | `wepr(...)` — default | `epr(...)` |
|---|---|---|
| Features | 2 × `k` | 1 |
| Reads | Each rank separately | Overall confidence per token |
| Needs calibrating | Yes | Yes |
| Choose it when | Almost always | Too few labelled examples to fit `2k` coefficients |

Both accept the same arguments and return the same thing, so switching is a one-word
change.

### Score every token

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

### Score a batch

Pass a list to score many responses in one call. They are parsed, padded to a common
length and scored together, one output row per generated sequence in input order:

```python
scores = detector.predict_proba([response_a, response_b])  # (n_sequences, 2)
```

Each response is validated on its own, so one malformed or too-narrow response is reported
by position rather than quietly changing its neighbours' scores.

### Score Langfuse traces

Fetch a trace, score its output, and write the probability back as a trace score. Re-runs
overwrite rather than duplicate, so this is safe to schedule:

```python
from artefactual.adapters.langfuse.evaluator import HallucinationEvaluator

evaluator = HallucinationEvaluator("wepr", langfuse_client, wepr("microsoft/phi-4"))
evaluator.score_trace(trace_id)
```

See `docs/examples/langfuse_integration_demo.ipynb`.

### Calibrate for your own model

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

### Use it inside scikit-learn

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

## Requirements

Artefactual reads a signal that has to be present in the response, so two things are
required of whatever produced it:

1. **`logprobs` and `top_logprobs` are enabled.** Providers that do not expose them cannot
   be scored at all.
2. **`top_logprobs` is at least `k`**, the rank count the calibration was fit at — 15 for
   every shipped file. See [The `k` parameter](#the-k-parameter) for what happens
   otherwise, and why narrower responses are refused rather than adjusted.

## Limitations

Worth knowing before you adopt it:

- **You need `logprobs` and `top_logprobs`.** Providers that do not expose them cannot be
  scored at all. `top_logprobs` must be at least the calibrated `k` (15), which also rules
  out providers capping it lower.
- **Calibrations are model-specific.** A calibration maps one model's confidence onto a
  probability. Four are shipped (see [Supported models](#supported-models));
  scoring a different model needs [your own calibration](#calibrate-for-your-own-model),
  which needs labelled data.
- **The probability is only as good as its calibration.** Rankings transfer more readily
  than absolute values — if you only need to triage the riskiest answers, the raw ordering
  is the robust part.
- **It measures uncertainty, not truth.** A model that is confidently wrong — a memorised
  misconception, a poisoned fine-tune — is not uncertain, and scores low. This is a
  complement to retrieval grounding or a judge, not a replacement for them.
- **Scoring is per sequence.** There is no cross-response consistency check, which is the
  signal sampling-based detectors buy with their extra generations.

## How it works

Each detector is a three-step scikit-learn pipeline:

```
LogProbParser  ->  EntropyTransformer  ->  PretrainedLogisticRegression
  response            uncertainty              calibrated probability
                       features
```

**1. Parse.** `LogProbParser` normalises OpenAI Chat Completions and Responses API payloads
— as SDK objects or plain dicts, singly or batched — into a dense
`(n_sequences, n_tokens, k)` array of log-probabilities, `NaN`-padded.

**2. Measure.** The middle step turns the raw distribution into a small feature vector
summarising how uncertain the model was. The shipped detectors measure this with the
entropy contribution of each candidate, `s_kj = -p_kj * log(p_kj)` for token *j* and rank
*k*, reduced two ways:

| Reduction | Feature vector | Reads |
|---|---|---|
| `epr` | `mean_j( mean_k s_kj )` — 1 feature | Mean contribution per rank, per token |
| `wepr` | `mean_j(s_kj)` ‖ `max_j(s_kj)` — 2k features | Per-rank mean and peak, weighted separately |

EPR averages over exactly `k` ranks, which is why `k` belongs to the feature's definition
and why responses carrying fewer ranks are rejected rather than padded. WEPR weights each
rank separately, which pays off because `-p*log(p)` peaks at `p = 1/e`: a mid-ranked
candidate carries more signal than either the near-certain top rank or the negligible tail.

**3. Calibrate.** A `LogisticRegression` pre-loaded with per-model coefficients maps the
features to a probability. Because it is a real sklearn classifier, `predict`,
`predict_proba` and `decision_function` all work as expected.

The measurement step is a plain transformer, so any callable can replace the reduction:

```python
import numpy as np

from artefactual.scoring import EntropyTransformer

EntropyTransformer(reduction=lambda s, axis: np.nanmax(s, axis=axis))
```

## Reference

### Supported models

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

### Input formats

Both OpenAI wire formats are accepted, as SDK objects or plain mappings. A minimal
Responses API payload looks like:

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
