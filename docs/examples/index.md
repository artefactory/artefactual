# Examples

Six runnable notebooks. Three of them — {doc}`epr_usage_demo`, {doc}`wepr_usage_demo` and
{doc}`train_wepr` — read committed fixtures rather than calling a model, so they need no
GPU and no API key. The first two still fetch their detector's weights from the Hugging
Face Hub; {doc}`train_wepr` fits its own and is the one that runs entirely offline.

**Training a detector for your own model?** Start with {doc}`train_wepr_pipeline` if you
need to produce the answers, or {doc}`train_wepr` if you already have them.
{doc}`train_wepr_bertjudge` starts where {doc}`train_wepr` does — from responses you
already have — but produces the verdicts itself, with `artefactory/BERTJudge`: a 210M
encoder that grades an answer against a reference and is efficient enough to run on CPU, so
labelling costs no API requests. It reads only the gold answer and not the alias list, and
explains itself with a number rather than a sentence.

| Notebook | Shows | Needs |
|---|---|---|
| {doc}`epr_usage_demo` | EPR scoring at sequence and token level, against a published detector | Network, for the detector |
| {doc}`wepr_usage_demo` | WEPR with the risky spans highlighted token by token | Network, for the detector |
| {doc}`train_wepr` | Fitting a WEPR detector on answers and verdicts you already have, and tuning it with scikit-learn | `matplotlib`, for the two figures |
| {doc}`train_wepr_pipeline` | Producing those answers and verdicts: your questions, generation with logprobs, an LLM judge | `[adapters]`, a `logprobs`-capable endpoint |
| {doc}`train_wepr_bertjudge` | Labelling responses you already have with a 210M encoder judge, then fitting on the pair | `bert-judge`, `torch`, `transformers>=4.57,<5`, a 420 MB judge download |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces through `HallucinationEvaluator` | `[adapters]`, a `logprobs`-capable endpoint, a Langfuse project |

Run one locally from the repository root:

```bash
uv run jupyter lab docs/examples/epr_usage_demo.ipynb
```

Outputs are committed and the documentation build does not re-execute them
(`nbsphinx_execute = "never"`), so the published pages stay reproducible offline.
`tests/test_examples.py` runs the notebooks against the current source, so a published
example cannot silently stop working — it checks the code, not the numbers beside it.

{doc}`langfuse_integration_demo` and {doc}`train_wepr_pipeline` generate against a live
endpoint, and {doc}`train_wepr_bertjudge` downloads a 420 MB judge, so all three ship
without stored outputs and are checked statically rather than executed — the numbers are
the ones your own run produces.

{doc}`train_wepr_pipeline` writes the two file *formats* {doc}`train_wepr` reads, under
its own names, so the two compose once you point the second at the first's output. Both are
the OpenAI Batch output shape, which is what `scripts/train_detector.py` reads as well:
produce once, refit as often as you like.

{doc}`train_wepr_bertjudge` reads the same responses {doc}`train_wepr` does and writes the
other half itself, in the same `judgments.jsonl` format — the verdict object is the
pipeline's contract, not any one API's, so an encoder judge conforms to it like a generative
one and `scripts/train_detector.py` reads either without knowing which wrote it.

{doc}`train_wepr`'s own answers and log-probabilities are **synthetic**; it opens by
saying so.

```{toctree}
:maxdepth: 1
:hidden:

epr_usage_demo
wepr_usage_demo
train_wepr
train_wepr_pipeline
train_wepr_bertjudge
langfuse_integration_demo
```
