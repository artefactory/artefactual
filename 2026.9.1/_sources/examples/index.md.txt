# Examples

Three runnable examples, each scoring with a detector rather than building one. Training
one for your own model is in {doc}`../guide/training`.

| Notebook | Shows | Needs |
|---|---|---|
| {doc}`epr_usage_demo` | EPR scoring at sequence and token level, against a published detector | Network, for the detector |
| {doc}`wepr_usage_demo` | WEPR with the risky spans highlighted token by token | Network, for the detector |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces through `HallucinationEvaluator` | `[adapters]`, a `logprobs`-capable endpoint, a Langfuse project |

The first two read a committed fixture rather than calling a model, so they need no GPU and
no API key; they still fetch their detector's weights from the Hugging Face Hub.

Run one locally from the repository root. The `notebooks` group holds what they need
beyond the package — `matplotlib`, the adapters, and the encoder judge's runtime — and
`jupytext` turns the Markdown into a notebook to open:

```bash
uv sync --group notebooks
uv run jupytext --to ipynb docs/examples/epr_usage_demo.md
uv run jupyter lab docs/examples/epr_usage_demo.ipynb
```

Every page also offers that notebook ready-made, as a download or in Colab.

The source of an example is its `.md`: code and prose, no stored outputs and no execution
counts, so a change to one reads as a diff. The published pages carry outputs because the
release build runs the notebooks and publishes what it got — which is also why a page you
are reading on a pull request preview shows code with nothing under it.

`tests/test_examples.py` runs them against the current source, so a published example
cannot silently stop working.

{doc}`langfuse_integration_demo` needs a live endpoint and a Langfuse project, so the build
never runs it: its cells are the record of a session you run yourself.

```{toctree}
:maxdepth: 1
:hidden:

epr_usage_demo
wepr_usage_demo
langfuse_integration_demo
```
