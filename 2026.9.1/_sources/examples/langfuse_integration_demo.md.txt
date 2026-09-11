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

# Langfuse Adapter Demo

This notebook demonstrates how to use the **HallucinationEvaluator** with the sklearn compatible **EPR** and **WEPR** detectors to score LLM traces in Langfuse.

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
    | `TOP_LOGPROBS` | no | Ranks to request per token (default 15) |
    | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | yes | Langfuse project |

    The endpoint must return `logprobs` with `top_logprobs`, otherwise there is nothing to score.

    `TOP_LOGPROBS` is passed to the detectors as `k`. Every shipped WEPR weights file is
    calibrated at 15 ranks, so raising it will make `WEPR()` raise — deliberately, since the
    coefficient vector is fixed at its calibration rank count. Either leave it at 15 or
    supply weights calibrated at the value you choose. `EPR()` is unaffected: an EPR
    calibration is a single coefficient, and `k` only governs how the rank axis is aligned. The defaults below target the HuggingFace router, where `OPENAI_API_KEY` is a token from your HuggingFace account.

```{code-cell} ipython3
# From a clone: `uv sync --group notebooks`.
#
# On Colab, uncomment to install the package.
# !pip install -q 'artefactual[adapters]'
```

## Run a generation and send it to Langfuse

```{code-cell} ipython3
import os
import time

from langfuse import get_client, observe
from langfuse.openai import OpenAI
from openai.types.chat import ChatCompletion

from artefactual.adapters.langfuse.evaluator import HallucinationEvaluator
from artefactual.scoring import EPR, WEPR, BaseDetector

# --- Configuration ---------------------------------------------------------
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # no default: it is a credential
TOP_LOGPROBS = int(os.environ.get("TOP_LOGPROBS", "15"))

# Both factories resolve a registry model name; pass a path instead to use your
# own calibration. K must match the rank count requested above.
# The detectors trained for OPENAI_MODEL, not OPENAI_MODEL itself: `EPR()` and `WEPR()`
# resolve a repository holding `model.skops`, and a generator's repository has none. The
# two reductions are published separately, so there are two ids. Swap both when you swap
# the model being scored -- a detector reads one model's confidence and no other's.
EPR_DETECTOR = "artefactory/epr-ministral"
WEPR_DETECTOR = "artefactory/wepr-ministral"
# ---------------------------------------------------------------------------

client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


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

langfuse = get_client()
langfuse.flush()

print("Waiting for Langfuse server to index the logprobs of the trace...")
time.sleep(3.0)

traces_to_evaluate = langfuse.api.trace.list(limit=1).data
```

## Score traces with EPR

A detector always needs weights: `EPR.from_pretrained` and `WEPR.from_pretrained` take a path to a weights file or a Hugging Face repository id. Constructing `EPR(...)` or `WEPR(...)` gives an *unfitted* detector instead, for training one of your own.

```{code-cell} ipython3
evaluator = HallucinationEvaluator(
    name="EPR",
    langfuse_client=langfuse,
    detector=EPR.from_pretrained(EPR_DETECTOR, k=TOP_LOGPROBS),
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
    detector=WEPR.from_pretrained(WEPR_DETECTOR, k=TOP_LOGPROBS),
)

for trace in traces_to_evaluate:
    score = evaluator.score_trace(trace.id)
    print(f"WEPR Scored Trace : {trace.id} → {score}")

langfuse.flush()
```
