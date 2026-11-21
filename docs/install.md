# Installation

Artefactual needs Python 3.11, 3.12 or 3.13 (`requires-python = ">=3.11,<3.14"`). The
floor comes from scikit-learn 1.8, the oldest release that reads the published detectors.

## From PyPI

```bash
pip install artefactual
```

or, in a uv project:

```bash
uv add artefactual
```

The base install pulls in `numpy`, `scikit-learn`, `skops`, `pydantic`, `beartype`
and `huggingface-hub`. That is everything needed to load a detector and score
responses.

## Extras

| Extra | Install | Adds |
|---|---|---|
| `adapters` | `pip install "artefactual[adapters]"` | `langfuse` and `openai`, for scoring Langfuse traces with `HallucinationEvaluator` |
| `docs` | `pip install "artefactual[docs]"` | Sphinx, MyST-NB and the theme, to build this site |

## From source

```bash
git clone https://github.com/artefactory/artefactual.git
cd artefactual
uv sync                # the package and the development tools
uv sync --extra docs   # also the documentation toolchain
```

The version is derived from the git tags, so install from a clone that has them rather
than from a source archive.

## Check the install

```bash
python -c "import artefactual; print(artefactual.__version__)"
```

Loading a detector downloads its weights from the Hugging Face Hub on first use and
reads them from the local cache afterwards:

```python
from artefactual import WEPR

K = 15  # the k the published detectors were trained with
detector = WEPR.from_pretrained("artefactory/wepr-ministral", k=K)
detector.k  # 15
```

Next: {doc}`guide/scoring` to score a response, or {doc}`guide/sklearn` to use a detector
inside scikit-learn tooling.
