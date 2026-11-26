---
file_format: mystnb
jupytext:
  notebook_metadata_filter: mystnb,file_format
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: artefactual (3.13.5)
  language: python
  name: python3
mystnb:
  execution_mode: 'off'
---

# Score Langfuse traces with a detector

Generate one answer through an OpenAI-compatible endpoint, trace it in Langfuse, and attach an EPR and a WEPR score to the trace with `HallucinationEvaluator`.

**Needs / cost:** a Langfuse project, an endpoint that returns `logprobs`, and the keys below; one generation request, then each detector's weights are downloaded once. The build never runs this notebook, so its cells are the record of a session you run yourself.

| Section | What it does | Cost |
|---|---|---|
| Prerequisites | Keys, endpoint and detector names | the work: a Langfuse project and a `logprobs`-capable endpoint |
| Run a generation and send it to Langfuse | One generation, traced | 1 request |
| Score traces with EPR | Attach an EPR score to the trace | the detector's weights, once |
| Score traces with WEPR | The same with WEPR | the detector's weights, once |

## Prerequisites

* Install the adapter dependencies:
    ```bash
    uv pip install artefactual[adapters]
    ```

* Create a free LangFuse project at [cloud.langfuse.com](https://cloud.langfuse.com), then go to **Settings → API Keys**.

* Set the following environment variables:

    | Variable | Required | Purpose |
    | --- | --- | --- |
    | `OPENAI_API_KEY` | yes | Credential for the endpoint |
    | `OPENAI_BASE_URL` | no | Any OpenAI-compatible endpoint |
    | `OPENAI_MODEL` | no | Model to generate with |
    | `TOP_LOGPROBS` | no | Candidates to request per token, the width `k` (default 15) |
    | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | yes | Langfuse project |

    The endpoint must return `logprobs` with `top_logprobs`, otherwise there is nothing to score.

    `TOP_LOGPROBS` is passed to the detectors as `k`; the rule for choosing it is in {doc}`../guide/scoring`. The defaults below target the HuggingFace router, where `OPENAI_API_KEY` is a token from your HuggingFace account.

```{code-cell} ipython3
:tags: [hide-input]

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys

# Colab starts from a runtime with neither the package nor the files that sit beside
# this notebook in the repository. Everywhere else -- a clone synced with
# `uv sync --group notebooks`, the docs build, the test suite -- both are already there,
# so this cell does nothing and there is nothing for a reader to uncomment.
ON_COLAB = "google.colab" in sys.modules

PACKAGES = [
    "artefactual[adapters]",
]

if ON_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]
```

## Run a generation and send it to Langfuse

```{code-cell} ipython3
:tags: [hide-input]

import getpass
import os
import time

from langfuse import get_client, observe
from langfuse.openai import OpenAI
from openai.types.chat import ChatCompletion

from artefactual import EPR, WEPR, make_detector
from artefactual.adapters.langfuse.evaluator import HallucinationEvaluator
```

Every knob in one place. `OPENAI_API_KEY` has no default because it is a credential.

The detector ids name the detectors trained **for** `OPENAI_MODEL`, not the model itself:
`EPR.from_pretrained` and `WEPR.from_pretrained` resolve a repository holding `model.skops`,
and a generator's repository has none. The two reductions are published separately, so there
are two ids. Swap both when you swap the model being scored — a detector reads one model's
confidence and no other's. `TOP_LOGPROBS` must match the width (candidates per token) each was trained at.

```{code-cell} ipython3
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
# The OpenAI and Langfuse clients read their keys from the environment.
for key in ("OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    if not os.environ.get(key):
        os.environ[key] = getpass.getpass(f"{key}: ")
TOP_LOGPROBS = int(os.environ.get("TOP_LOGPROBS", "15"))

EPR_DETECTOR = "artefactory/epr-ministral"
WEPR_DETECTOR = "artefactory/wepr-ministral"
```

```{code-cell} ipython3
:tags: [hide-input]

# `langfuse.openai.OpenAI` is the stock client with Langfuse's instrumentation wrapped
# around it, so every call it makes becomes a trace without another line of code.
client = OpenAI(base_url=OPENAI_BASE_URL)
```

`@observe()` is what puts the generation on a trace. `logprobs=True` and
`top_logprobs=TOP_LOGPROBS` are what make that trace scorable: without them Langfuse records
the text and the detector has nothing to read.

```{code-cell} ipython3
@observe()
def run_generation() -> ChatCompletion:
    return client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )


print("Generated message:", run_generation().choices[0].message.content)
```

```{code-cell} ipython3
:tags: [hide-input]

# Traces are sent in the background, so the fetch below has to wait for the server to have
# indexed this one -- otherwise the evaluators score an empty list.
langfuse = get_client()
langfuse.flush()

print("Waiting for Langfuse server to index the logprobs of the trace...")
time.sleep(3.0)

traces_to_evaluate = langfuse.api.trace.list(limit=1).data
```

## Score traces with EPR

A detector always needs weights: `EPR.from_pretrained` and `WEPR.from_pretrained` take a path to a weights file or a Hugging Face repository id. Constructing `EPR(...)` or `WEPR(...)` gives an *unfitted* classifier instead, for training one of your own. `HallucinationEvaluator` takes a pipeline from responses to probabilities, so `make_detector(k=..., classifier=...)` puts a `LogProbParser` in front of each detector.

```{code-cell} ipython3
evaluator = HallucinationEvaluator(
    name="EPR",
    langfuse_client=langfuse,
    detector=make_detector(k=TOP_LOGPROBS, classifier=EPR.from_pretrained(EPR_DETECTOR, k=TOP_LOGPROBS)),
)

for trace in traces_to_evaluate:
    score = evaluator.score_trace(trace.id)
    print(f"EPR Scored Trace : {trace.id} → {score}")

langfuse.flush()
```

## Score traces with WEPR

```{code-cell} ipython3
evaluator = HallucinationEvaluator(
    name="WEPR",
    langfuse_client=langfuse,
    detector=make_detector(k=TOP_LOGPROBS, classifier=WEPR.from_pretrained(WEPR_DETECTOR)),
)

for trace in traces_to_evaluate:
    score = evaluator.score_trace(trace.id)
    print(f"WEPR Scored Trace : {trace.id} → {score}")

langfuse.flush()
```

## Where to go next

- **Scoring without Langfuse.** {doc}`epr_usage_demo` and {doc}`wepr_usage_demo` score
  responses directly.
- **A detector for your own model.** {doc}`train_wepr` fits one; pass it to
  `HallucinationEvaluator` in place of the published ones above.
