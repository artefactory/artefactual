# Examples

Five runnable notebooks. The first three read committed fixtures, so they need no GPU, API
key or model download.

| Notebook | Shows | Needs |
|---|---|---|
| {doc}`epr_usage_demo` | EPR scoring at sequence and token level, on a fixture narrower than the rank count the weights were trained at | Nothing |
| {doc}`wepr_usage_demo` | WEPR at its trained rank count, with the risky spans highlighted token by token | Nothing |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces through `HallucinationEvaluator` | `[adapters]`, a `logprobs`-capable endpoint, a Langfuse project |
| {doc}`train_wepr` | Fitting a WEPR detector on 50 already-labelled answers shipped alongside it | Nothing |
| {doc}`train_wepr_pipeline` | The whole procedure for your own model — questions, generation, judging, training | `[adapters]`, `datasets`, a `logprobs`-capable endpoint |

Run one locally from the repository root:

```bash
uv run jupyter lab docs/examples/epr_usage_demo.ipynb
```

Outputs are committed and the documentation build does not re-execute them
(`nbsphinx_execute = "never"`), so the published pages stay reproducible offline.
`tests/test_examples.py` runs the notebooks against the current source, which is what keeps
those stored outputs honest.

{doc}`langfuse_integration_demo` and {doc}`train_wepr_pipeline` generate against a live
endpoint, so they ship without stored outputs and are checked statically rather than
executed — the numbers are the ones your own run produces.

{doc}`train_wepr` reads `wepr_training_sample.json`, whose answers and labels are
**synthetic**: the file says so, and so does the notebook. It exists to exercise the
training path offline, not to report a model's score.

```{toctree}
:maxdepth: 1
:hidden:

epr_usage_demo
wepr_usage_demo
langfuse_integration_demo
train_wepr
train_wepr_pipeline
```
