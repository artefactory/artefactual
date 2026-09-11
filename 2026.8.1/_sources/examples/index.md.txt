# Examples

Three runnable notebooks. The first two read committed response fixtures, so they need no
GPU, API key or model download.

| Notebook | Shows | Needs |
|---|---|---|
| {doc}`epr_usage_demo` | EPR scoring at sequence and token level, on a fixture narrower than the rank count the weights were trained at | Nothing |
| {doc}`wepr_usage_demo` | WEPR at its trained rank count, with the risky spans highlighted token by token | Nothing |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces through `HallucinationEvaluator` | `[adapters]`, a `logprobs`-capable endpoint, a Langfuse project |

Run one locally from the repository root:

```bash
uv run jupyter lab docs/examples/epr_usage_demo.ipynb
```

Outputs are committed and the documentation build does not re-execute them
(`nbsphinx_execute = "never"`), so the published pages stay reproducible offline.
`tests/test_examples.py` runs the notebooks against the current source, which is what keeps
those stored outputs honest.

```{toctree}
:maxdepth: 1
:hidden:

epr_usage_demo
wepr_usage_demo
langfuse_integration_demo
```
