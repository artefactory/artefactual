# Artefactual

Artefactual assigns a language model's answer a probability of being a hallucination. It
reads the answer that has already been generated, together with the token probabilities
returned alongside it, and needs nothing else from the model.

```bash
pip install artefactual
```

```python
from artefactual.scoring import BaseDetector

# response is what any OpenAI-compatible client returns with logprobs=True, top_logprobs=15
detector = BaseDetector.from_pretrained("artefactory/wepr-ministral", "wepr")
detector.predict_proba(response)[:, 1]   # P(hallucination) per sequence
```

Two answers from the same model, scored by the same detector:

| The model was asked | It answered | P(hallucination) |
|---|---|---|
| What is the capital city of France? | Paris. | 0.08 |
| Who is Charles Moslonka? Where was he born? | Charles Moslonka is a French singer born in Lyon in 1985. | 0.99 |

Neither answer was checked against anything. The second scores high because the model was
uncertain while generating it. Both numbers are the ones {doc}`examples/wepr_usage_demo`
prints, and it also marks which tokens drove them.

The [project README](https://github.com/artefactory/artefactual) covers installation,
requirements and the published results. This site covers using a detector in depth.

- {doc}`guide/scoring` — choosing a detector, thresholds, batches, Langfuse traces
- {doc}`guide/training` — fitting one for a model with no published weights, in three notebooks
- {doc}`guide/how-it-works` — the three pipeline stages and the two entropy reductions
- {doc}`guide/reference` — the rank count, weight-file layout, accepted response shapes
- {doc}`examples/index` — runnable notebooks: EPR and WEPR scoring, and Langfuse traces
- {doc}`api` — generated signatures

```{toctree}
:maxdepth: 2
:hidden:

guide/scoring
guide/training
guide/how-it-works
guide/reference
examples/index
api
presentations/index
```
