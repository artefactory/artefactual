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
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Artefactual Package Demo: Hallucination Detection with EPR

This notebook demonstrates the `artefactual` package for scoring LLM outputs, specifically focusing on hallucination detection using entropy-based methods. Here we will use EPR (Entropy Production Rate), which computes entropy at each token and averages it across the entire sequence.

We explore two examples:
1.  **General Knowledge Question**:

    *"What is the capital city of France?"* → `"Paris."` (expected high certainty, low entropy expected)

2.  **Hallucination Trigger**: Asking about the first author of our paper (Charles Moslonka) to observe how the model hallucinates biographical details.

    *"Who is Charles Moslonka?"* → a fabricated biography (expected high uncertainty, high entropy expected)

We will use:
* **JSON fixture (open_ai_responses_top15.json):** two mock OpenAI Responses API outputs, carrying 15 top logprobs per token -- the rank count the shipped calibrations were fit at.
* **Published detector:** named by its own Hugging Face repository (`artefactory/epr-ministral`), fetched on first use and cached. A path to a local `.skops` file is accepted the same way.
* **EPR:** scorer from the artefactual package via the scikit-learn pipeline API.
* **Visualizations:** each token highlighted by its own score, so the uncertain stretches of an answer are visible rather than inferred.

```{code-cell} ipython3
:tags: [hide-input]

# On Colab, uncomment to install the package and fetch the files this notebook reads.
# !pip install -q artefactual
# !wget -q https://raw.githubusercontent.com/artefactory/artefactual/main/docs/examples/open_ai_responses_top15.json
```

```{code-cell} ipython3
import json
import warnings
from pathlib import Path

from IPython.display import HTML, display
from sklearn.exceptions import InconsistentVersionWarning

from artefactual.scoring import EPR
```

```{code-cell} ipython3
# The Hugging Face repository holding the published EPR detector for the model that
# produced these responses. A path to your own `.skops` file works too.
DETECTOR = "artefactory/epr-ministral"
DATA_PATH = "open_ai_responses_top15.json"

# The rank count the detector was calibrated at, which the fixture also carries. Passing a
# different value raises rather than producing a mis-shaped score, and a response narrower
# than K is refused rather than padded: the missing ranks are unfetched, not absent, so
# filling them would understate the entropy.
K = 15

# Where the highlighting below changes colour.
THRESHOLD_LOW = 0.35  # at or below -> green, the model was confident here
THRESHOLD_HIGH = 0.70  # above -> red, it was not
```

## Load Example Responses

The fixture contains two responses, to illustrate the contrast between a certain and an
uncertain answer.

```{code-cell} ipython3
with Path(DATA_PATH).open(encoding="utf-8") as f:
    data = json.load(f)

responses = data["responses"]
print(f"Loaded {len(responses)} responses")
```

## Build the EPR Pipeline

The detector is fetched from the Hub on first use, then cached. `EPR.from_pretrained`
always requires a repository id or a path. The raw OpenAI Responses API dicts go straight
to the pipeline; parsing is its first step.

```{code-cell} ipython3
# The published weights were written by an older scikit-learn than the one installed here,
# which warns on unpickling. It is a note to whoever republishes the detector, not to the
# reader loading it, and the estimator it produces is the same either way.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

detector = EPR.from_pretrained(DETECTOR, k=K)
```

## Sequence-Level Scoring

`predict_proba(response)` returns an array of shape `(n_sequences, 2)`. Column 1 is the
hallucination probability: higher means the model was more uncertain while generating the
answer. Nothing here was checked against any source.

```{code-cell} ipython3
for resp in responses:
    prompt = resp["metadata"]["prompt"]
    text = resp["output"][0]["content"][0]["text"]
    score = detector.predict_proba(resp)[0, 1]

    print(f"Prompt : {prompt}")
    print(f"Answer : {text}")
    print(f"Score  : {score:.2f}")
    print()
```

## Token-Level Scoring

`predict_token_proba(response)` returns `(n_sequences, max_tokens, 1)`: the same
probability, per token, which is what says *where* in the answer the model was uncertain.


* **`n_sequences`**: response index.
* **`max_tokens`**: token index within the sequence.
* **`1`**: the per-token scalar hallucination probability.

```{code-cell} ipython3
:tags: [hide-input]

def color_for(score) -> str:
    """Green below THRESHOLD_LOW, red above THRESHOLD_HIGH, yellow in between."""
    if score <= THRESHOLD_LOW:
        return "rgba(0, 255, 0, 0.3)"
    if score <= THRESHOLD_HIGH:
        return "rgba(255, 255, 0, 0.3)"
    return "rgba(255, 0, 0, 0.3)"


def highlight(tokens, scores) -> HTML:
    """The answer as it was generated, each token on a background reading its own score."""
    spans = []
    for token, score in zip(tokens, scores, strict=True):
        shown = token.replace("\n", "<br>")
        spans.append(
            f'<span style="background-color: {color_for(score)}; padding: 2px; margin: 1px; '
            f'border-radius: 3px;">{shown}</span>'
        )
    return HTML('<div style="font-family: monospace; font-size: 14px; line-height: 1.5;">' + "".join(spans) + "</div>")
```

```{code-cell} ipython3
for resp in responses:
    prompt = resp["metadata"]["prompt"]
    tokens = [t["token"] for t in resp["output"][0]["content"][0]["logprobs"]]
    scores = detector.predict_token_proba(resp)[0, :, 0]

    print(f"Prompt: {prompt}")
    display(highlight(tokens, scores))
```

## Your model is not one of the published detectors?

A detector reads one model's confidence, so the four published ones only score the four
models they were trained on. `EPR(k=15).fit(responses, y)` fits your own, on that
model's answers and a verdict on each.
