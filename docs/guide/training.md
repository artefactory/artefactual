# Training a detector

Any model that returns `top_logprobs` can have a detector fitted for it, published weights
or not. It takes answers that model generated and a verdict on each. The pipeline is the one
{doc}`how-it-works` describes, unchanged — only the classifier's coefficients are new.

```python
# responses: what the model answered, generated with top_logprobs >= k
# y:         one 0/1 label per answer, 1 marking a hallucination
detector = WEPR(k=15).fit(responses, y)
```

A fit needs two things, both in the OpenAI Batch output shape and joined on `custom_id`:
the answers, and a verdict on each. Three notebooks fit a detector from them, each starting
from a different amount of that already in hand — none depends on the others, and each one
ends in a saved `.skops` file.

| You already have | Notebook |
|---|---|
| the answers, and a verdict on each | {doc}`../examples/train_wepr` |
| the answers, but nothing judged yet | {doc}`../examples/train_wepr_bertjudge` |
| an endpoint, and nothing else | {doc}`../examples/train_wepr_pipeline` |

Each runs top to bottom as it ships — the first two on committed sample files, the third on
a hundred TriviaQA questions — so the detector is fitted before anything is edited. Pointing
one at your own data is two file paths, or for the third, the question list in one cell.

## How much data

A hundred judged answers is enough to see whether the detector separates the two classes,
which is the scale the notebooks run at. The paper's own runs use TriviaQA at around 500
questions per model — the procedure is in
[`scripts/ecir`](https://github.com/artefactory/artefactual/tree/main/scripts/ecir). `epr`
fits one coefficient rather than `2k`, so it is the one to reach for when labels are scarcer
than that.

## The label a judge's verdict becomes

`read_judgment` returns `True` when the answer was **correct**, and the class a detector
predicts is the hallucination. The caller negates it, and that is the whole of the
conversion:

| The judge said | `read_judgment` | `y = int(not verdict)` | Means |
|---|---|---|---|
| the answer was correct | `True` | `0` | grounded |
| the answer was wrong | `False` | `1` | hallucination — the class `predict_proba[:, 1]` scores |
| something unreadable | `None` | — | drop the row and count it |

## Whether the fit worked

Each notebook cross-validates the fit and reports a ROC-AUC with a per-class breakdown, so
the answer is in the run itself rather than taken on trust. *What the fit weighs* in
{doc}`../examples/train_wepr` plots the fitted coefficient per rank, and *Audit the labels
the judge produced* in {doc}`../examples/train_wepr_bertjudge` lists, question by question,
where the judge and the fitted detector disagree — the labels are the part worth doubting
first.

```{toctree}
:maxdepth: 1
:hidden:

/examples/train_wepr
/examples/train_wepr_bertjudge
/examples/train_wepr_pipeline
```
