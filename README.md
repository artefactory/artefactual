# Artefactual

**Estimating how likely a language model answer is to be a hallucination.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![Paper](https://img.shields.io/badge/arXiv-2509.04492-b31b1b.svg)](https://arxiv.org/abs/2509.04492)

Artefactual is a Python module that assigns a language model's answer a probability of
being a hallucination. It reads the answer that has already been generated, together with
the token probabilities returned alongside it, and needs nothing else from the model.

## Quick start

```python
from openai import OpenAI

from artefactual.scoring import wepr

MODEL = "mistralai/Ministral-8B-Instruct-2410"

client = OpenAI(base_url="https://your-provider.example/v1")  # any OpenAI-compatible endpoint
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Who wrote the Rust book?"}],
    logprobs=True,
    top_logprobs=15,
)

detector = wepr(DETECTOR)
print(detector.predict_proba(response)[:, 1])   # P(hallucination) per sequence
print(detector.predict_token_proba(response))   # ...and per token
```

Weights are model-specific, so the name passed to `wepr` must be the model that produced
the response. There is no default threshold; choosing one is covered in the
[user guide](https://artefactory.github.io/artefactual/guide/scoring.html).

No endpoint is needed to try the library. The
[example notebooks](https://artefactory.github.io/artefactual/examples/) run against two
checked-in responses and need no GPU, API key or model download.

## Requirements

Two conditions apply to whatever produced the response:

1. **`logprobs` and `top_logprobs` are enabled.** Providers that do not expose them cannot
   be scored at all.
2. **`top_logprobs` is at least `k`**, the rank count the detector's own weights were
   trained at. The shipped files use `k = 15`.

Neither needs auditing in advance — a response that fails one is refused by name. Why a
narrow response is refused rather than padded is in
[the reference](https://artefactory.github.io/artefactual/guide/reference.html#the-k-parameter).

## Installation

```bash
pip install artefactual
```

### Dependencies

Artefactual requires:

- Python (>= 3.10)
- NumPy
- scikit-learn
- pydantic
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

Two detectors are provided. `wepr` is the default and the more accurate; `epr` fits a
single coefficient instead of `2k`, for when labelled data is scarce. Both take the same
arguments and return the same type.

```python
from artefactual.scoring import epr, wepr

wepr("chicham/artefactual-wepr-phi4")         # a published detector, by its own repo
wepr("/path/to/my_detector.skops")           # one you trained yourself
wepr(k=15, trainable=True).fit(responses, y)  # fit your own, y is 0/1 per sequence
epr("chicham/artefactual-epr-phi4")           # the single-coefficient variant
```

Scoring a batch, reading per-token scores, scoring Langfuse traces, composing into
`GridSearchCV` and training a detector for a model that is not shipped are covered in the
[user guide](https://artefactory.github.io/artefactual/guide/scoring.html).

## Published detectors

A detector is named by its own Hugging Face repository, not by the model it scores. Pick
the row for the model that produced the responses, and the column for the reduction:

| Model that produced the responses | `epr()` | `wepr()` |
|---|---|---|
| `mistralai/Ministral-8B-Instruct-2410` | `chicham/artefactual-epr-ministral` | `chicham/artefactual-wepr-ministral` |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | `chicham/artefactual-epr-mistral-small` | `chicham/artefactual-wepr-mistral-small` |
| `tiiuae/Falcon3-10B-Instruct` | `chicham/artefactual-epr-falcon3` | `chicham/artefactual-wepr-falcon3` |
| `microsoft/phi-4` | `chicham/artefactual-epr-phi4` | `chicham/artefactual-wepr-phi4` |

All are trained at `k = 15`. Both factories also accept a path to a `.skops` file, so a
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
