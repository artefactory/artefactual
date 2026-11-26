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

# Score responses with WEPR

Score two committed responses, one certain and one hallucinated, with a published WEPR detector, and highlight the risky tokens.

WEPR (Weighted Entropy Production Rate) learns one weight per rank, because the second-most-likely
candidate carries different information from the fifteenth.

The two questions are *"What is the capital city of France?"*, answered with certainty, and
*"Who is Charles Moslonka?"* (the first author of our paper), answered with a fabricated biography.

**Needs / cost:** no API key, no GPU; one detector download from the Hugging Face Hub (~KBs, cached), then seconds.

| Section | What it does | Cost |
|---|---|---|
| Load example responses | Read the committed fixture | none |
| Build the WEPR detector | Fetch the published detector's weights | one download, ~KBs, cached after the first run |
| Sequence-level scoring | P(hallucination) per response | seconds |
| Token-level scoring | P(hallucination) per token, highlighted | seconds |

```{code-cell} ipython3
:tags: [hide-input]

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import urllib.request

# Colab starts from a runtime with neither the package nor the files that sit beside
# this notebook in the repository. Everywhere else -- a clone synced with
# `uv sync --group notebooks`, the docs build, the test suite -- both are already there,
# so this cell does nothing and there is nothing for a reader to uncomment.
ON_COLAB = "google.colab" in sys.modules

PACKAGES = [
    "artefactual",
]

FETCH = [
    "https://raw.githubusercontent.com/artefactory/artefactual/main/docs/examples/open_ai_responses_top15.json",
]

if ON_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]
    for url in FETCH:
        urllib.request.urlretrieve(url, url.rsplit("/", 1)[-1])  # ruff: ignore[suspicious-url-open-usage]
```

```{code-cell} ipython3
:tags: [hide-input]

import json
import warnings
from pathlib import Path

from IPython.display import HTML, display
from sklearn.exceptions import InconsistentVersionWarning

# The published weights were written by an older scikit-learn than the one installed here,
# which warns on unpickling. It is a note to whoever republishes the detector, not to the
# reader loading it, and the estimator it produces is the same either way.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from artefactual import WEPR, make_detector
```

```{code-cell} ipython3
# The Hugging Face repository holding the published WEPR detector for the model that
# produced these responses. A path to your own `.skops` file works too.
DETECTOR = "artefactory/wepr-falcon3"
DATA_PATH = "open_ai_responses_top15.json"

# Candidates per token the detector was trained at, which the fixture also carries. A detector refuses
# arrays of a different width rather than producing a mis-shaped score, and a response narrower
# than K is refused rather than padded: the missing candidates are unfetched, not absent, so
# filling them would understate the entropy.
K = 15

# Where the highlighting below changes colour.
THRESHOLD_LOW = 0.35  # at or below -> green, the model was confident here
THRESHOLD_HIGH = 0.70  # above -> red, it was not
```

## Load example responses

The fixture contains two responses, to illustrate the contrast between a certain and an
uncertain answer.

```{code-cell} ipython3
with Path(DATA_PATH).open(encoding="utf-8") as f:
    data = json.load(f)

responses = data["responses"]
print(f"Loaded {len(responses)} responses")
```

## Build the WEPR detector

The detector is fetched from the Hub on first use, then cached. `WEPR.from_pretrained`
always requires a repository id or a path. `make_detector` puts a `LogProbParser` in front: one pipeline from the raw OpenAI Responses API dicts
to scores.

```{code-cell} ipython3
detector = make_detector(k=K, classifier=WEPR.from_pretrained(DETECTOR))
```

## Sequence-level scoring

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

## Token-level scoring

`predict_token_proba` returns `(n_sequences, max_tokens, 1)`: the same
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

## Where to go next

- **Your model is not one of the published detectors?** A detector reads one model's
  confidence, so a published one only scores the model it was trained on. Fit your own with
  `make_detector(k=15).fit(responses, y)`: see {doc}`train_wepr`.
- **Score live traces.** {doc}`langfuse_integration_demo` attaches a detector's score to
  Langfuse traces.
- **The other detector.** {doc}`epr_usage_demo` runs the same fixture through EPR.
