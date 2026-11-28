# Examples

Six runnable examples: two score responses with a published detector, three train a
detector of your own, and one scores live Langfuse traces. Installation is in
{doc}`../install` and the scikit-learn behaviour the notebooks rely on is in
{doc}`../guide/sklearn`; the training workflow is in {doc}`../guide/training`.

| Notebook | Shows | Needs |
|---|---|---|
| {doc}`epr_usage_demo` | EPR scoring at sequence and token level, against a published detector | Network, for the detector |
| {doc}`wepr_usage_demo` | WEPR with the risky spans highlighted token by token | Network, for the detector |
| {doc}`train_wepr` | Training a detector from answers that are already judged | Nothing: it runs on the shipped fixtures |
| {doc}`train_wepr_lightweight_judge` | Training from unjudged answers, graded by a local encoder judge | A ~420 MB judge download, once |
| {doc}`train_wepr_pipeline` | Training from an endpoint alone: generate, judge, fit | An endpoint returning `top_logprobs`, and its API key |
| {doc}`langfuse_integration_demo` | Scoring live Langfuse traces through `HallucinationEvaluator` | `[adapters]`, a `logprobs`-capable endpoint, a Langfuse project |

Run one locally with `uv sync --group notebooks`, then `uv run jupytext --to ipynb
docs/examples/<name>.md`; every page also offers its notebook as a download or in Colab.
The pipeline and Langfuse notebooks need a live endpoint, so the build never runs them.

```{toctree}
:maxdepth: 1
:hidden:

epr_usage_demo
wepr_usage_demo
train_wepr
train_wepr_lightweight_judge
train_wepr_pipeline
langfuse_integration_demo
```
