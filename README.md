# Artefactual

**Estimating how likely a language model answer is to be a hallucination.**

[![Tests](https://github.com/artefactory/artefactual/actions/workflows/tests.yaml/badge.svg)](https://github.com/artefactory/artefactual/actions/workflows/tests.yaml)
[![Docs](https://github.com/artefactory/artefactual/actions/workflows/docs.yml/badge.svg)](https://github.com/artefactory/artefactual/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/artefactual.svg)](https://pypi.org/project/artefactual/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![Paper](https://img.shields.io/badge/arXiv-2509.04492-b31b1b.svg)](https://arxiv.org/abs/2509.04492)

Artefactual is a Python module that assigns a language model's answer a probability of
being a hallucination. It reads the answer that has already been generated, together with
the token probabilities returned alongside it, and needs nothing else from the model.

A *response* is the result of a call to an OpenAI-compatible API (chat completions, Responses
API or batch output) made with `logprobs=True` and `top_logprobs`.

The classifiers and the entropy transformer are scikit-learn estimators, checked with
`parametrize_with_checks`; the detectors, which read responses, are the same estimators with
a parser in front. See the
[scikit-learn compatibility guide](https://artefactory.github.io/artefactual/guide/sklearn.html).

## Quick start

```python
from openai import OpenAI

from artefactual import WEPR

MODEL = "mistralai/Ministral-8B-Instruct-2410"  # the model being scored
DETECTOR = "artefactory/wepr-ministral"  # the detector trained for it

client = OpenAI(base_url="https://your-provider.example/v1")  # any OpenAI-compatible endpoint
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Who wrote the Rust book?"}],
    logprobs=True,
    top_logprobs=15,
)

detector = WEPR.from_pretrained(DETECTOR, k=15)
print(detector.predict_proba(response)[:, 1])  # P(hallucination) per sequence
print(detector.predict_token_proba(response))  # ...and per token
```

Column order follows `detector.classes_`: with `0`/`1` labels, column 1 is the hallucination class. With other labels, pick the column with `list(detector.classes_).index(label)`.

A detector is trained for one model, so `DETECTOR` must be the one published for whatever
produced the response — see [published detectors](#published-detectors) for the pairs. There is no default threshold; choosing one is covered in the
[user guide](https://artefactory.github.io/artefactual/guide/scoring.html).

No endpoint is needed to try the library: two of the
[example notebooks](https://artefactory.github.io/artefactual/examples/) run against
checked-in fixtures and need no GPU and no API key. Training a detector for a model with no
published weights is covered, with three more notebooks, in
[training a detector](https://artefactory.github.io/artefactual/guide/training.html).

## Requirements

Two conditions apply to whatever produced the response:

1. **`logprobs` and `top_logprobs` are enabled.** Providers that do not expose them cannot
   be scored at all.
2. **`top_logprobs` is at least `k`**, the width the detector's own weights were
   trained at (candidates per token). The shipped files use `k = 15`.

Neither needs auditing in advance — a response that fails one is refused by name. Why a
narrow response is refused rather than padded is in
[the reference](https://artefactory.github.io/artefactual/guide/reference.html#the-k-parameter).

## Installation

```bash
pip install artefactual
```

### Dependencies

Artefactual requires:

- Python (>= 3.11)
- NumPy
- scikit-learn (>= 1.8)
- pydantic
- skops
- huggingface-hub
- beartype

Two optional extras are available: `[adapters]` installs `langfuse` and `openai` for the
integration examples, and `[docs]` installs Sphinx and the theme for building the
documentation.

### From source

```bash
git clone https://github.com/artefactory/artefactual
cd artefactual
uv sync
```

## Usage

Two detectors are provided. `WEPR` fits `2k` coefficients and is the more accurate; `EPR`
fits a single coefficient instead, for when labelled data is scarce. Both take the same
arguments and return the same type. They score completion responses; `EPRClassifier` and
`WEPRClassifier` are the same estimators over logprob arrays you already hold.

```python
from artefactual import EPR, WEPR, WEPRClassifier

# Unfitted, as a scikit-learn estimator is; y is 0/1 per sequence, 1 marking a hallucination
WEPR(k=15).fit(responses, y)

# Published weights, or your own file, are loaded rather than fitted
WEPR.from_pretrained("artefactory/wepr-phi4", k=15)
WEPR.from_pretrained("/path/to/my_detector.skops", k=15)
EPR.from_pretrained("artefactory/epr-phi4", k=15)  # the single-coefficient variant

# Logprob arrays, shaped (n_sequences, n_tokens, k) or one token per row (n, k)
WEPRClassifier(k=15).fit(logprobs, y)
```

Scoring a batch, reading per-token scores, scoring Langfuse traces, composing into
`GridSearchCV` and training a detector for a model that is not shipped are covered in the
[user guide](https://artefactory.github.io/artefactual/guide/scoring.html).

## Published detectors

A detector is named by its own Hugging Face repository, not by the model it scores. Pick
the row for the model that produced the responses, and the column for the pooling:

| Model that produced the responses | `EPR` | `WEPR` |
|---|---|---|
| `mistralai/Ministral-8B-Instruct-2410` | `artefactory/epr-ministral` | `artefactory/wepr-ministral` |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `artefactory/epr-mistral-small` | `artefactory/wepr-mistral-small` |
| `tiiuae/Falcon3-10B-Instruct` | `artefactory/epr-falcon3` | `artefactory/wepr-falcon3` |
| `microsoft/phi-4` | `artefactory/epr-phi4` | `artefactory/wepr-phi4` |

All are trained at `k = 15`. Both classes also accept a path to a `.skops` file, so a
detector you trained yourself is named the same way one published here is — the package
holds no list of models, and publishing another detector needs no release.

## Limitations

- **Not every provider can be scored.** The requirements above rule out any provider that
  hides `logprobs`, and any that caps `top_logprobs` below the detector's `k`.
- **Detectors are model-specific.** Scoring a model that is not shipped requires training a
  detector for it, which requires labelled data.
- **The probability is only as good as its training data.** Rankings transfer more readily
  than absolute values.
- **It measures uncertainty, not truth.** A model that is confidently wrong is not
  uncertain, and scores low. This complements retrieval grounding or a judge rather than
  replacing either.
- **Scoring is per sequence.** There is no cross-response consistency check.

## Results

ROC-AUC on TriviaQA hallucination detection at `k = 15`, as reported in Table 1 of the
[paper](https://arxiv.org/abs/2509.04492):

| Model | SelfCheckGPT | EPR | HalluDetect | WEPR |
|---|---|---|---|---|
| `Mistral-Small-3.1-24B` | 79.0 | 74.6 | 78.7 | **82.0** |
| `Falcon-3-10B` | 70.1 | 75.4 | 79.0 | **84.1** |
| `Phi-4` (14.7B) | 71.4 | 78.2 | 83.8 | **85.4** |
| `Ministral-8B-2410` | 81.1 | 81.4 | **86.1** | 85.8 |

The full tables and the procedure that produced them are in the
[`scripts/ecir`](https://github.com/artefactory/artefactual/tree/main/scripts/ecir)
subdirectory.

## Documentation

- Documentation and user guide: https://artefactory.github.io/artefactual/
- Example notebooks: https://artefactory.github.io/artefactual/examples/
- Training a detector: https://artefactory.github.io/artefactual/guide/training.html
- How it works: https://artefactory.github.io/artefactual/guide/how-it-works.html

## Development

Contributions are welcome — see
[CONTRIBUTING.md](https://github.com/artefactory/artefactual/blob/main/CONTRIBUTING.md).

- Source code: https://github.com/artefactory/artefactual
- Issue tracker: https://github.com/artefactory/artefactual/issues

```bash
uv sync
uv run pytest tests        # test suite (uv sync resolves the env first)
uv run pytest tests --cov  # with coverage

uvx ruff check src tests   # lint — a standalone tool, no project env needed
uvx ruff format src tests
uvx --from shellcheck-py shellcheck scripts/ecir/*.sh
```

## Versioning

Releases are dated (`YYYY.MM.PATCH`). Breaking changes to the public surface are called out
in the [release notes](https://github.com/artefactory/artefactual/releases).

## Citation

If `artefactual` is useful in your research, please cite the ECIR 2026 paper:

```bibtex
@inproceedings{moslonka2026learned,
  title     = {Learned Hallucination Detection in Black-Box LLMs Using Token-Level Entropy Production Rate},
  author    = {Moslonka, Charles and Randrianarivo, Hicham and Garnier, Arthur and Malherbe, Emmanuel},
  booktitle = {Advances in Information Retrieval},
  series    = {Lecture Notes in Computer Science},
  volume    = {16483},
  pages     = {115--130},
  publisher = {Springer, Cham},
  year      = {2026},
  doi       = {10.1007/978-3-032-21289-4_8},
}
```

## License

MIT — no limitation of usage, including for commercial applications.
