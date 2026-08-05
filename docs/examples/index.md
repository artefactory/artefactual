# Examples

Runnable notebooks covering the two detectors and the Langfuse integration. The scoring
demos need no GPU, no API key and no model download — they read committed fixtures of
OpenAI responses, so they run offline:

```bash
uv run jupyter lab docs/examples/epr_usage_demo.ipynb
```

The Langfuse notebook is the exception: it generates live, so it needs the `adapters`
extra, an OpenAI-compatible endpoint that returns `logprobs`, and a Langfuse project.

```{toctree}
:maxdepth: 1

epr_usage_demo
wepr_usage_demo
langfuse_integration_demo
```

## What each one shows

| Notebook | Shows |
|---|---|
| {doc}`epr_usage_demo` | Sequence- and token-level scoring with EPR, on a fixture whose rank count differs from the calibration's |
| {doc}`wepr_usage_demo` | WEPR at its calibrated rank count, plus token-level highlighting of the risky spans |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces with `HallucinationEvaluator` |

The notebooks are committed with their outputs and the documentation build does not
re-execute them (`nbsphinx_execute = "never"`), so the site stays reproducible offline.
`tests/test_examples.py` executes them against the current source, which is what keeps
those stored outputs honest.
