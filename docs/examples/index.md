# Examples

Five runnable notebooks. Three of them — {doc}`epr_usage_demo`, {doc}`wepr_usage_demo` and
{doc}`train_wepr` — read committed fixtures, so they need no GPU, API key or model
download.

**Training a detector for your own model?** Start with {doc}`train_wepr_pipeline` if you
need to produce the answers, or {doc}`train_wepr` if you already have them.

| Notebook | Shows | Needs |
|---|---|---|
| {doc}`epr_usage_demo` | EPR scoring at sequence and token level, on a fixture narrower than the rank count the weights were trained at | Nothing |
| {doc}`wepr_usage_demo` | WEPR at its trained rank count, with the risky spans highlighted token by token | Nothing |
| {doc}`train_wepr_pipeline` | Producing training data for your own model: questions, generation with logprobs, an LLM judge | `[adapters]`, `datasets`, a `logprobs`-capable endpoint |
| {doc}`train_wepr` | Fitting a WEPR detector on answers and verdicts you already have | Nothing |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces through `HallucinationEvaluator` | `[adapters]`, a `logprobs`-capable endpoint, a Langfuse project |

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
executed — the numbers are the ones your own run produces. The pipeline writes the three
file *formats* {doc}`train_wepr` reads — two of them the OpenAI Batch output shape — under
its own names, so the two compose once you point the second at the first's output, and
`scripts/train_detector.py` reads the same two: produce once, refit as often as you like.

{doc}`train_wepr`'s three shipped files are in the OpenAI Batch output shape, so they need
no conversion in either direction — `scripts/train_detector.py` reads the same two. Its
answers and log-probabilities are **synthetic**; the notebook opens by saying so.

```{toctree}
:maxdepth: 1
:hidden:

epr_usage_demo
wepr_usage_demo
langfuse_integration_demo
train_wepr
train_wepr_pipeline
```
