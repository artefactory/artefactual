# Examples

Four runnable notebooks. Three of them — {doc}`epr_usage_demo`, {doc}`wepr_usage_demo` and
{doc}`train_wepr` — read committed fixtures, so they need no GPU, API key or model
download.

| Notebook | Shows | Needs |
|---|---|---|
| {doc}`epr_usage_demo` | EPR scoring at sequence and token level, on a fixture narrower than the rank count the weights were trained at | Nothing |
| {doc}`wepr_usage_demo` | WEPR at its trained rank count, with the risky spans highlighted token by token | Nothing |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces through `HallucinationEvaluator` | `[adapters]`, a `logprobs`-capable endpoint, a Langfuse project |
| {doc}`train_wepr` | Validating answers you already have and fitting a WEPR detector on them | Nothing |

Run one locally from the repository root:

```bash
uv run jupyter lab docs/examples/epr_usage_demo.ipynb
```

Outputs are committed and the documentation build does not re-execute them
(`nbsphinx_execute = "never"`), so the published pages stay reproducible offline.
`tests/test_examples.py` runs the notebooks against the current source, which is what keeps
those stored outputs honest.

{doc}`langfuse_integration_demo` generates against a live endpoint, so it ships without
stored outputs and is checked statically rather than executed.

{doc}`train_wepr` reads three shipped files, joined on `custom_id`:
`questions_sample.json` for the questions, and `responses_sample.jsonl` +
`judgments_sample.jsonl` in the OpenAI Batch output shape — the same two files
`scripts/train_detector.py` reads, so they need no conversion in either direction. The 100
questions and gold answers are real facts, the first a real TriviaQA row; **the answers
and their log-probabilities are synthetic** — no model produced them, and the verdicts
were computed from them. The notebook says so, and exists to show the validation and
training path, not to report a model's score.

```{toctree}
:maxdepth: 1
:hidden:

epr_usage_demo
wepr_usage_demo
langfuse_integration_demo
train_wepr
```
