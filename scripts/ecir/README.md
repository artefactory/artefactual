# Reproducing the ECIR EPR / WEPR experiments

Trains an EPR or WEPR detector from scratch: generate answers, have an LLM grade them,
train the detector on the result, and check how well it separates hallucinations. This is
the procedure from *"Learned Hallucination Detection in Black-Box LLMs using Token-level
Entropy Production Rate"*, and the same one to follow for training a detector on a model of your own.

**The two LLM stages — generating the answers, and grading them — are run by
[`vllm run-batch`](https://docs.vllm.ai/en/stable/examples/offline_inference/openai_batch.html).**
The scripts here only prepare its input and read its output; nothing in this repo calls a
model.

## What you need

| | Why | Where |
|---|---|---|
| `vllm` | Runs the two LLM stages | A Linux GPU box — it has no macOS wheels |
| `jq` | Builds the batch request files | Anywhere |
| This repo, `uv sync`'d | Trains and evaluates the detector | Anywhere |
| `questions.json` | Your QA pack — see below | You write this |

Only steps 2 and 4 need the GPU. The rest runs on a laptop, so the usual split is to build
the request files locally, copy them over, run the batches, and copy the outputs back.

## The training data you need

The detector is a logistic regression trained on `(response, was_it_a_hallucination)` pairs.
You supply the questions and their gold answers; the pipeline produces the responses, and
the LLM judge produces the labels.

So the only file you write is `questions.json`, a list of:

```json
[{"question": "Who sent Augustine to England?",
  "question_id": "q-1",
  "short_answer": "Pope Gregory",
  "answer_aliases": ["Gregory I", "the Pope"]}]
```

| Field | Required | Used for |
|---|---|---|
| `question` | yes | The prompt sent to the model being calibrated |
| `question_id` | yes | Becomes `custom_id`; every stage joins on it, so it must be unique |
| `short_answer` | yes | The gold answer the judge grades against |
| `answer_aliases` | no | Other answers the judge should accept; omit or leave empty |

The paper uses **TriviaQA** for training and **WebQuestions** to test generalisation, plus
a financial RAG corpus (ArGiMi-Ardian) for missing-context detection. Any short-form QA set
works, including your own domain questions. Two properties matter:

- **Answers must be short enough to grade automatically.** The judge compares against
  `short_answer`; an essay cannot be scored this way.
- **The set must be hard enough that the model gets some wrong.** The fit needs both
  classes. A model that answers everything correctly produces no hallucinations to learn
  from, and the fit will fail with a single-class error.

A few hundred questions is a workable start. Step 6 reports a 95% interval; if it is too
wide to tell EPR and WEPR apart, that is the signal to label more.

## Tutorial

Seven steps, start to finish. Set these once — `K` in particular has to be the same number
in steps 1 and 6, and mismatching it is the most common way to train a detector that scores wrongly:

```bash
cd scripts/ecir
MODEL=mistralai/Ministral-8B-Instruct-2410   # the model you are calibrating
JUDGE=mistralai/Ministral-8B-Instruct-2410   # the model that grades it
K=15                                          # ranks per token; every shipped file uses 15
OUT=out
mkdir -p "$OUT"
```

### Step 1 — build the generation requests

```bash
./build_generation_requests.sh questions.json "$MODEL" "$K" > "$OUT/gen_requests.jsonl"
```

One JSON line per question, in the OpenAI batch format, asking for `top_logprobs: $K`.
Check it before spending GPU time:

```bash
head -1 "$OUT/gen_requests.jsonl" | jq '{custom_id, model: .body.model, k: .body.top_logprobs}'
wc -l < "$OUT/gen_requests.jsonl"   # should equal your question count
```

Sampling follows the paper (§4.1.2): non-greedy decoding at `T_samp = 1.0`, `top_p = 1.0`,
sampling cutoff `K_samp = 50`. Override with `GEN_TEMPERATURE`, `GEN_TOP_P`, `GEN_TOP_K`.

The 200-token cap is this script's default, not the paper's — the paper only notes that the
tasks yield short answers. The paper does report that dropping to `T_samp = 0.6` changed
ROC-AUC by less than 1 point on Falcon-3-10B, so the signal is not an artefact of sampling
hot.

### Step 2 — generate the answers *(GPU)*

```bash
vllm run-batch -i "$OUT/gen_requests.jsonl" -o "$OUT/responses.jsonl" --model "$MODEL"
```

This is the expensive step. Confirm every request came back with the right rank width:

```bash
jq -r 'select(.response != null)
       | (.response.body // .response).choices[0].logprobs.content[0].top_logprobs | length' "$OUT/responses.jsonl" | sort -u
```

One number should print, and it must equal `$K`. If it is smaller, the generation ignored
`top_logprobs` — fix that and rerun, because step 6 will refuse the data.

### Step 3 — build the judge requests

```bash
./build_judge_requests.sh questions.json "$OUT/responses.jsonl" "$JUDGE" > "$OUT/judge_requests.jsonl"
```

Joins each generated answer back to its gold answer on `custom_id` and renders the paper's
grading prompt. Generations that failed are dropped, and the count is reported on stderr.

### Step 4 — grade the answers *(GPU)*

```bash
vllm run-batch -i "$OUT/judge_requests.jsonl" -o "$OUT/judgments.jsonl" --model "$JUDGE"
```

The judge replies with `{"judgment": true|false, "explanation": "..."}`. `true` means the
answer was **correct**, so the training label is its negation — 1 marks a hallucination.
Spot-check a few:

```bash
jq -r 'select(.response != null) | (.response.body // .response).choices[0].message.content' "$OUT/judgments.jsonl" | head -3
```

### Step 5 — check the class balance

Before fitting, make sure you have both classes:

```bash
jq -r 'select(.response != null) | (.response.body // .response).choices[0].message.content' "$OUT/judgments.jsonl" \
  | grep -c '"judgment": *true'
```

Compare against your question count. All-correct or all-wrong cannot be fit — go back to
step 1 with harder or easier questions.

### Step 6 — train the detector and evaluate it

```bash
uv run ../train_detector.py \
    --responses "$OUT/responses.jsonl" \
    --judgments "$OUT/judgments.jsonl" \
    --reduction wepr --k "$K" \
    --output "$OUT/wepr_weights.json" \
    --report "$OUT/wepr_evaluation.json"
```

One script, because the two halves are one question. It splits the labelled set once
(stratified, `--test_size`, default 0.25), fits on the training part, writes the weights,
then scores that fitted detector on the part it never saw. **The numbers describe the file
in `--output`** — what is reported is what is shipped, not an average over models that were
refitted and discarded.

`uv run` resolves the project environment itself, so there is no activation step and no
chance of picking up a different install. Numbers here are illustrative:

```
joined 400 pairs on custom_id (112 hallucinations)
fitting on 300 response(s), holding out 100
intercept: -3.02
wrote out/wepr_weights.json

               precision    recall  f1-score   support

    grounded       0.88      0.93      0.90        72
hallucination      0.74      0.61      0.67        28

    accuracy                           0.84       100

   roc_auc: 0.7412  [0.6810, 0.7955]
    pr_auc: 0.5233  [0.4401, 0.6118]
```

Read the two halves as different questions. ROC-AUC and PR-AUC score the *ranking*, which
is what a detector is for when you triage by score; the classification report scores the
decisions at the 0.5 threshold. A detector can rank well and still make poor calls there,
so read both — and if you intend to act on a different threshold, the AUCs are the ones
that carry over. Recall on the `hallucination` row is usually the number that matters: the
fraction of hallucinations actually flagged.

Repeat with `--reduction epr` for the other detector and compare. Steps 2 and 4 do not need
repeating — both detectors are fit from the same generations.

Pass `--test_size 0` to fit on everything and skip the evaluation. Do that only once you
have already measured the recipe and want the last few percent of data in the model.

### Step 7 — use the trained detector

```python
from artefactual.scoring import wepr

detector = wepr("out/wepr_weights.json", k=15)
detector.predict_proba(response)[:, 1]
```

The file is the same format `load_weights` reads, so it is interchangeable with the shipped
ones. Score responses generated the same way — same model, and `top_logprobs` at least `k`.

## What the paper reports

ROC-AUC at `K = 15`, from Table 1 of the paper. Higher is better; **bold** is the best of
the four methods on that row.

### TriviaQA — hallucination detection

| Model | SelfCheckGPT | EPR | HalluDetect | WEPR |
|---|---|---|---|---|
| `Mistral-Small-3.1-24B` | 79.0 | 74.6 | 78.7 | **82.0** |
| `Falcon-3-10B` | 70.1 | 75.4 | 79.0 | **84.1** |
| `Phi-4` (14.7B) | 71.4 | 78.2 | 83.8 | **85.4** |
| `Ministral-8B-2410` | 81.1 | 81.4 | **86.1** | 85.8 |

### WebQuestions — generalisation (detectors trained on TriviaQA)

| Model | SelfCheckGPT | EPR | HalluDetect | WEPR |
|---|---|---|---|---|
| `Mistral-Small-3.1-24B` | 59.3 | 62.5 | 62.8 | **64.8** |
| `Falcon-3-10B` | 65.8 | 68.2 | 69.3 | **73.2** |
| `Phi-4` (14.7B) | 65.0 | 65.2 | 66.3 | **66.6** |
| `Ministral-8B-2410` | 66.2 | 65.4 | 71.6 | **72.6** |

Two things worth reading off these: **WEPR beats EPR on every row**, which is why `wepr` is
the default, and the absolute numbers drop by 10-20 points when the detector meets a dataset
it was not trained on. Train on data that resembles your traffic.

SelfCheckGPT needs 10 extra generations per answer for those numbers; EPR and WEPR need
none. The paper measures roughly 80 ± 20 µs per score against at least 10 s for
SelfCheckGPT.

## Use k = 15

`k` is the number of ranks per token, and it appears in two places that **must agree**:
`build_generation_requests.sh` (step 1, where it becomes `top_logprobs`) and `--k` on
`train_detector.py` (step 6). Every shipped detector was trained at 15. Setting `K` once at
the top of the tutorial is what keeps them in step.

It cannot be inferred, because it is part of the metric: EPR is the entropy of the top `k`
ranks (Eq. 3 and 6 of the paper), so changing `k` changes what is being measured, and WEPR
has one coefficient per rank.

Generating with a smaller `top_logprobs` than you fit at fails loudly, for both reductions,
as soon as the responses are read:

```
ValueError: Response 0 carries 5 rank(s) per token but k=15 was requested. The missing
ranks are not absent from the distribution, only unfetched, so zero-filling them would drop
their entropy contributions and score the response as more confident than it was.
Regenerate with top_logprobs=15, or score at k=5 with a detector trained at that rank count.
```

Generating *wider* is harmless — surplus ranks are dropped — so when in doubt, request more.
Supplying WEPR weights whose rank count disagrees with `--k` is caught separately, by their
coefficient vector:

```
ValueError: Weights cover 15 rank(s) but k=20 was requested. WEPR coefficients are fixed
at the rank count used during calibration; pass k=15, or supply weights calibrated at k=20.
```

If you generated at a narrower `k`, regenerate — do not reuse a detector trained at another.
## If something looks wrong

**`Response N carries M rank(s) per token but k=15 was requested`.** The generation batch
was produced with a smaller `top_logprobs` than you are fitting at — most often because
step 1 and step 6 were run with different `k`. Check what the responses actually carry:

```bash
jq -r 'select(.response != null)
       | (.response.body // .response).choices[0].logprobs.content[0].top_logprobs | length' out/responses.jsonl | sort -u
```

One number should come back, and it must be at least your `--k`. If it is smaller, rerun
steps 1 and 2 — the judgments are unaffected and do not need regenerating.

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
jq -r 'select(.response != null) | (.response.body // .response).choices[0].message.content' out/judgments.jsonl | head
```

**`No custom_id is present in both files`.** The two files came from different batches.
`custom_id` round-trips from `question_id` through both `run-batch` calls, so every stage
joins by id — a reordered or partially failed batch can never pair a generation with the
wrong verdict, but two unrelated batches will not join at all.

**`5-fold cross-validation needs at least 5 of each class` from the evaluation.** The rarer
class — usually the hallucinations — has fewer members than there are folds, so it cannot
appear in every one. The message prints the actual counts. Label more data, or lower
`--folds`; note that fewer folds means a noisier estimate, so treat it as a way to get a
reading at all rather than a fix.

## What the scripts read

`vllm run-batch` writes one line per request:

```json
{"id": "vllm-383d...", "custom_id": "q-1",
 "response": {"status_code": 200, "request_id": "vllm-batch-be0f...",
              "body": {"choices": [{"message": {...}, "logprobs": {"content": [...]}}]}},
 "error": null}
```

This is the OpenAI Batch output spec: `response` is an envelope, and the ChatCompletion is
its `body`. Steps 3, 6 and 7 unwrap it themselves, so there is no conversion step.

Older vllm put the completion directly in `response`, with no envelope. The scripts accept
either, so batch files produced before the change still read — but note the difference when
inspecting a file by hand, because `jq '.response.choices[0]'` silently yields `null` on a
current file rather than failing.

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

`train_detector.py` splits the labelled set once with `train_test_split`, stratified so the
hallucination rate is the same on both sides, fits the detector on the training part, and
scores it on the held-out part. **The evaluation is of the model that was written to
`--output`** — the detector is fitted once and never refitted, so the numbers belong to the
artefact you keep rather than to the recipe that produced it.

`--test_size` (default 0.25) sets the holdout; `--seed` (default 42) fixes both the split
and the resampling, so a rerun reproduces the run exactly. `--test_size 0` fits on
everything and skips the evaluation, which is only worth doing once the recipe is already
measured.

The confidence intervals come from resampling the *held-out predictions* with replacement
(`--repetitions`, default 1000) and taking the 2.5th and 97.5th percentiles. Resamples that
come back single-class are dropped, because ROC-AUC is undefined on them; the surviving
count is reported next to the interval and is worth reading when hallucinations are rare.

**How this differs from the paper.** The paper bootstraps the whole procedure — resample,
*refit*, score what fell out, repeat — so its interval measures how much the fitting varies
across datasets. This resamples one fixed model's scores, which answers the narrower
question of how precisely that model's ROC-AUC is known. The intervals here are therefore
tighter than Table 1's, and the two are not directly comparable. The point estimate is
comparable; the spread is not.

Alongside the AUCs, `classification_report` gives per-class precision, recall and F1 at the
0.5 threshold. Read them as different questions: the AUCs score the *ranking*, which is
what matters when you triage by score, while the report scores the decisions. Recall on the
`hallucination` row is usually the number to watch — the fraction actually flagged.

## Not reproduced

The SelfCheckGPT and HalluDetect baselines the paper compares against are not part of the
EPR/WEPR method and are not included here, so the paper's two comparison columns cannot be
rebuilt from this repo.
