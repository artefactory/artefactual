# Reproducing the ECIR EPR / WEPR experiments

Rebuilds the EPR and WEPR calibrations from
*"Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate"*
and reports their out-of-bag bootstrap scores.

**The two LLM stages — generating the answers, and rating them with an LLM judge — are run
by [`vllm run-batch`](https://docs.vllm.ai/en/stable/examples/offline_inference/openai_batch.html).**
The scripts here only prepare its input and read its output; nothing in this repo calls a
model.

## Before you start

| You need | Why |
|---|---|
| `vllm` on `PATH`, on a Linux GPU box | Runs the two batches. Not an `artefactual` dependency, and it has no macOS wheels. |
| `jq` | Builds the batch request files. |
| `artefactual` installed (`uv sync`) | Fits and evaluates the calibrations. |
| `questions.json` | Your QA pack (below). |

Only the two `run-batch` stages need the GPU box; everything else runs anywhere.

`questions.json` is a list of:

```json
[{"question": "Who sent Augustine to England?",
  "question_id": "q-1",
  "short_answer": "Pope Gregory",
  "answer_aliases": ["Gregory I", "the Pope"]}]
```

`question_id` must be unique — it becomes the key every later stage joins on.

## Run it

```bash
./run_pipeline.sh questions.json out/ \
    mistralai/Ministral-8B-Instruct-2410 \
    mistralai/Ministral-8B-Instruct-2410 \
    15
```

Arguments are `questions.json`, output directory, generator model, judge model, and `k`
(the number of top logprobs to request per token — **use 15**, see below).

You end up with, in `out/`:

| File | What it is |
|---|---|
| `responses.jsonl` | Generated answers with their top-15 logprobs (`vllm run-batch` output) |
| `judgments.jsonl` | The judge's verdicts (`vllm run-batch` output) |
| `{epr,wepr}_weights.json` | The fitted calibrations, in the format `load_weights` reads |
| `{epr,wepr}_evaluation.json` | Mean ROC-AUC / PR-AUC with 95% intervals |

To use a fitted calibration:

```python
from artefactual.scoring import wepr

detector = wepr("out/wepr_weights.json", k=15)
detector.predict_proba(response)[:, 1]
```

## Or stage by stage

Useful when the GPU box and your workstation are different machines: run steps 2 and 4
there, the rest anywhere.

```bash
MODEL=mistralai/Ministral-8B-Instruct-2410
JUDGE=mistralai/Ministral-8B-Instruct-2410

# 1. build the generation requests
./build_generation_requests.sh questions.json "$MODEL" 15 > out/gen_requests.jsonl

# 2. generate  (LLM)
vllm run-batch -i out/gen_requests.jsonl -o out/responses.jsonl --model "$MODEL"

# 3. build the judge requests -- reads the answers from step 2
./build_judge_requests.sh questions.json out/responses.jsonl "$JUDGE" > out/judge_requests.jsonl

# 4. rate  (LLM)
vllm run-batch -i out/judge_requests.jsonl -o out/judgments.jsonl --model "$JUDGE"

# 5. fit -- reads steps 2 and 4
python ../train_calibration.py --responses out/responses.jsonl --judgments out/judgments.jsonl \
    --reduction wepr --k 15 --output out/wepr_weights.json

# 6. evaluate -- reads steps 2 and 4
python ../evaluate_calibration.py --responses out/responses.jsonl --judgments out/judgments.jsonl \
    --reduction wepr --k 15 --output out/wepr_evaluation.json
```

Repeat steps 5 and 6 with `--reduction epr` for the EPR row. Steps 2 and 4 do not need
repeating — both detectors are fit from the same generations.

## Use k = 15

`k` appears in three places and **must be the same number in all of them**: the third
argument to `build_generation_requests.sh` (it becomes `top_logprobs` in the request), and
`--k` on both Python scripts. Every shipped calibration uses 15.

It cannot be inferred, because it is part of the metric: EPR averages the entropy
contribution over the top `k` ranks, so changing `k` rescales the score, and WEPR has one
coefficient per rank. Getting it wrong fails loudly rather than silently for WEPR:

```
ValueError: Weights cover 15 rank(s) but k=20 was requested. WEPR coefficients are fixed
at the rank count used during calibration; pass k=15, or supply weights calibrated at k=20.
```

If you generate at a different `k`, regenerate — do not reuse a calibration fit at another.

## If something looks wrong

**`joined N pairs on custom_id` reports fewer than your question count.** Some requests
failed. Lines whose request errored carry `"error"` and a null `"response"`; they are
dropped and counted rather than crashing the run. Check the count in the log and inspect:

```bash
jq -c 'select(.error != null) | {custom_id, error}' out/responses.jsonl
```

**`dropped N verdict(s) that could not be parsed`.** The judge is asked for
`{"judgment": true|false, "explanation": "..."}` and returned something else. Inspect a
few and, if the model is simply chatty, raise `JUDGE_MAX_TOKENS`:

```bash
jq -r 'select(.response != null) | .response.choices[0].message.content' out/judgments.jsonl | head
```

**`No custom_id is present in both files`.** The two files came from different batches.
`custom_id` round-trips from `question_id` through both `run-batch` calls, so every stage
joins by id — a reordered or partially failed batch can never pair a generation with the
wrong verdict, but two unrelated batches will not join at all.

**`No out-of-bag fold held two classes` from the evaluation.** Too few examples, or too
few of one class, for any bootstrap fold to hold both. Label more data.

## What the scripts read

`vllm run-batch` writes one line per request:

```json
{"id": "vllm-383d...", "custom_id": "q-1",
 "response": {"choices": [{"message": {...}, "logprobs": {"content": [...]}}]},
 "error": null}
```

`response` is the ChatCompletion itself — it is *not* nested under `response.body`. Steps
3, 5 and 6 consume this directly; there is no conversion step.

## Prompts

`prompts/generate.txt` and `prompts/judge.txt` hold the paper's prompts verbatim.
Placeholders are substituted by `jq` with literal split/join, so a question containing
backslashes, `&` or quotes cannot corrupt the rendering.

`prompts/judge.jinja` is the original jinja2 template, kept so the rendering can be
re-checked against it — it was verified byte-identical for 0, 1 and 2 aliases. Edit the
prompts and you are no longer reproducing the paper.

The judge's `judgment: true` means the answer was correct, so the training label is its
negation: **1 marks a hallucination**.

## Evaluation method

`evaluate_calibration.py` reproduces the paper's out-of-bag bootstrap: resample the
labelled set with replacement, fit on the sample, score whatever fell out of it, repeat
1000 times (`--repetitions`). It reports the mean the paper quotes plus a 95% percentile
interval — the mean alone cannot say whether a gap between two detectors is real.

Resampling uses `sklearn.utils.resample` seeded per repetition (`--seed`, default 42), so
a rerun reproduces the same folds. Folds whose out-of-bag set holds fewer than two
examples or only one class are skipped and counted, because ROC-AUC is undefined there.

## Not reproduced

The SelfCheckGPT and Halludetect baselines are not part of the EPR/WEPR method and are not
included, so the paper's baseline comparison columns cannot be rebuilt from this repo.
