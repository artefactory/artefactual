# Examples

Three runnable notebooks, each scoring with a detector rather than building one. Training
one for your own model is in {doc}`../guide/training`.

| Notebook | Shows | Needs |
|---|---|---|
| {doc}`epr_usage_demo` | EPR scoring at sequence and token level, against a published detector | Network, for the detector |
| {doc}`wepr_usage_demo` | WEPR with the risky spans highlighted token by token | Network, for the detector |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces through `HallucinationEvaluator` | `[adapters]`, a `logprobs`-capable endpoint, a Langfuse project |

The first two read a committed fixture rather than calling a model, so they need no GPU and
no API key; they still fetch their detector's weights from the Hugging Face Hub.

Run one locally from the repository root. The `notebooks` group holds what they need beyond
the package — `matplotlib`, the adapters, and the encoder judge's runtime:

```bash
uv sync --group notebooks
uv run jupyter lab docs/examples/epr_usage_demo.ipynb
```

Every page also offers the notebook itself, as a download or in Colab.

Outputs are committed and the documentation build does not re-execute them
(`nbsphinx_execute = "never"`), so the published pages stay reproducible offline.
`tests/test_examples.py` runs the notebooks against the current source, so a published
example cannot silently stop working — it checks the code, not the numbers beside it.

{doc}`langfuse_integration_demo` generates against a live endpoint, so it ships without
stored outputs and is checked statically rather than executed — the numbers are the ones
your own run produces.

```{toctree}
:maxdepth: 1
:hidden:

epr_usage_demo
wepr_usage_demo
langfuse_integration_demo
```
