# Artefactual

Artefactual assigns a language model's answer a probability of being a hallucination. It
reads the answer that has already been generated, together with the token probabilities
returned alongside it, and needs nothing else from the model.

```bash
pip install artefactual
```

```python
from artefactual.scoring import BaseDetector

detector = BaseDetector.from_pretrained("artefactory/wepr-ministral", "wepr")
detector.predict_proba(response)[:, 1]   # P(hallucination) per sequence
```

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
