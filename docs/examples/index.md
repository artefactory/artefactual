# Examples

Six runnable notebooks. Three of them — {doc}`epr_usage_demo`, {doc}`wepr_usage_demo` and
{doc}`train_wepr` — read committed fixtures rather than calling a model, so they need no
GPU and no API key. The first two still fetch their detector's weights from the Hugging
Face Hub; {doc}`train_wepr` fits its own and is the one that runs entirely offline.

The package reads results and never produces them: a completion an API returned, or a line
of a Batch output file. Generating those results is the caller's business —
{doc}`train_wepr_pipeline` shows one way of doing it, against any OpenAI-compatible
endpoint.

**Training a detector for your own model?** Pick by what you already have.

| You have | Start with | What you change |
|---|---|---|
| An endpoint, and nothing else | {doc}`train_wepr_pipeline` | nothing — it runs on a hundred TriviaQA questions out of the box |
| Your own questions, each with a gold answer | {doc}`train_wepr_pipeline` | one cell, the question list; the rest is unchanged |
| Responses with `top_logprobs`, and a verdict on each | {doc}`train_wepr` | two file paths |
| Responses with `top_logprobs`, but nothing judged yet | {doc}`train_wepr_bertjudge` | two file paths; it writes the verdicts itself |

The last one judges with `artefactory/BERTJudge`: a 210M encoder that grades an answer
against a reference and is efficient enough to run on CPU, so labelling costs no API
requests. It reads only the gold answer and not the alias list, and explains itself with a
number rather than a sentence.

| Notebook | Shows | Needs |
|---|---|---|
| {doc}`epr_usage_demo` | EPR scoring at sequence and token level, against a published detector | Network, for the detector |
| {doc}`wepr_usage_demo` | WEPR with the risky spans highlighted token by token | Network, for the detector |
| {doc}`train_wepr` | Fitting a WEPR detector on answers and verdicts you already have, and tuning it with scikit-learn | `matplotlib`, for the two figures |
| {doc}`train_wepr_pipeline` | Producing those answers and verdicts: your questions, generation with logprobs, an LLM judge | `[adapters]`, a `logprobs`-capable endpoint |
| {doc}`train_wepr_bertjudge` | Labelling responses you already have with a 210M encoder judge, then fitting on the pair | `bert-judge`, `torch`, `transformers>=4.57,<5`, a 420 MB judge download |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces through `HallucinationEvaluator` | `[adapters]`, a `logprobs`-capable endpoint, a Langfuse project |

Run one locally from the repository root. The `notebooks` group holds what they need
beyond the package — `matplotlib`, the adapters, and the encoder judge's runtime:

```bash
uv sync --group notebooks
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
