# Reference

Detail behind the rank count a detector is pinned to, the format of a weights file,
and the response shapes the parser accepts.

## The `k` parameter

`k` is the top-k rank count used for scoring. It defaults to **15**, the rank count every
shipped file was trained at.

```python
detector = wepr("chicham/artefactual-wepr-phi4", k=15)
```

**Responses must carry at least `k` ranks.** This is an input requirement rather than
something the pipeline reconciles: generation must set `top_logprobs` to `k` or higher. The
two directions are not symmetric:

- **Wider than `k`** — surplus ranks are dropped. The detector never saw them, and a
  mean over `k` ranks is defined without them, so scoring is unaffected.
- **Narrower than `k`** — refused. Those ranks are not absent from the distribution, only
  unfetched, so filling them with zeros drops their contributions from the sum:

  ```
  ValueError: Response 0 carries 5 rank(s) per token but k=15 was requested. The missing
  ranks are not absent from the distribution, only unfetched, so zero-filling them would
  drop their entropy contributions and score the response as more confident than it was.
  Regenerate with top_logprobs=15, or score at k=5 with a detector trained at that rank
  count.
  ```

  The resulting score is wrong rather than merely rescaled — a narrow response can score
  as *more* confident than a genuinely wider one — so it is rejected instead of adjusted.
  Each response is checked individually, so a wide response cannot mask a narrow one in
  the same batch.

WEPR additionally validates the weights themselves, since its coefficient vector has one
entry per rank:

```
ValueError: Weights cover 15 rank(s) but k=20 was requested. WEPR coefficients are
fixed at the rank count they were trained at; pass k=15, or supply weights
trained at k=20.
```

EPR detectors record no rank count, so for `epr()` the `k` passed governs the input width.

## Detector files

A locally trained detector is loaded by passing a path instead of a name:

```python
detector = wepr("/path/to/my_detector.skops")
```

Weight files are JSON. EPR files carry a single coefficient:

```json
{"intercept": -2.91, "coefficients": {"mean_entropy": 58.17}}
```

WEPR weights carry a `mean_rank_i` and `max_rank_i` pair for each rank `1..k`:

```json
{"intercept": -3.02, "coefficients": {"mean_rank_1": 3.81, "max_rank_1": -0.44}}
```

## Input formats

Both OpenAI wire formats are accepted, as SDK objects or plain mappings. A minimal
Responses API payload looks like:

```python
response = {
    "object": "response",
    "output": [
        {
            "content": [
                {
                    "logprobs": [
                        {"top_logprobs": [{"logprob": -0.1}, {"logprob": -2.3}]},
                        {"top_logprobs": [{"logprob": -0.05}, {"logprob": -3.1}]},
                    ]
                }
            ]
        }
    ],
}
```

Malformed input fails loudly: a payload carrying no logprobs raises `TypeError`, and a
positive or non-finite log-probability raises `ValueError` naming the offending sample,
token and rank.

The generated signatures for every public object are in the
[API reference](https://artefactory.github.io/artefactual/api.html).
