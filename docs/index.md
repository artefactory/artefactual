# Artefactual

Artefactual assigns a language model's answer a probability of being a hallucination. It
reads the answer that has already been generated, together with the token probabilities
returned alongside it, and needs nothing else from the model.

```bash
pip install artefactual
```

```python
from artefactual import WEPR

# response: the result of an OpenAI-compatible call made with logprobs=True and top_logprobs
detector = WEPR.from_pretrained("artefactory/wepr-ministral", k=15)
detector.predict_proba(response)[:, 1]  # P(hallucination) per sequence
```

Column order follows `detector.classes_`: with `0`/`1` labels, column 1 is the hallucination class. With other labels, look the column up as {doc}`guide/sklearn` shows.

Two answers from the same model, scored by the same detector:

| The model was asked | It answered | P(hallucination) |
|---|---|---|
| What is the capital city of France? | Paris. | 0.08 |
| Who is Charles Moslonka? Where was he born? | Charles Moslonka is a French singer born in Lyon in 1985. | 0.99 |

Neither answer was checked against anything. The second scores high because the model was
uncertain while generating it. Both numbers are the ones {doc}`examples/wepr_usage_demo`
prints, and it also marks which tokens drove them.

A *response* here is the result of a call to an OpenAI-compatible API (chat completions,
Responses API or batch output) made with `logprobs` and `top_logprobs`.

The [project README](https://github.com/artefactory/artefactual) covers the requirements
and the published results. This site covers installing and using a detector in depth.

- {doc}`install` — installing from PyPI or from source
- {doc}`guide/scoring` — choosing a detector, thresholds, batches, Langfuse traces
- {doc}`guide/training` — fitting one for a model with no published weights, in three notebooks
- {doc}`guide/how-it-works` — the three pipeline stages and the two entropy poolings
- {doc}`guide/reference` — candidates and ranks, weight-file layout, accepted inputs
- {doc}`guide/sklearn` — scikit-learn compatibility, model selection, feature names, persistence
- {doc}`examples/index` — runnable notebooks: EPR and WEPR scoring, and Langfuse traces
- {doc}`api` — generated signatures
- {doc}`changelog` — what changed in each release
- {doc}`contributing`

```{toctree}
:maxdepth: 2
:hidden:

install
guide/scoring
guide/training
guide/how-it-works
guide/reference
guide/sklearn
examples/index
api
presentations/index
changelog
contributing
```
