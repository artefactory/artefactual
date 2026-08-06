# Reproducing the ECIR EPR / WEPR experiments

Trains an EPR or WEPR detector from scratch: generate answers, have an LLM grade them,
train the detector on the result, and check how well it separates hallucinations. This is
the procedure from *"Learned Hallucination Detection in Black-Box LLMs using Token-level
Entropy Production Rate"*, and the procedure to follow for training a detector on any other model.

**The two LLM stages — generating the answers, and grading them — are run by
[`vllm run-batch`](https://docs.vllm.ai/en/stable/examples/offline_inference/openai_batch.html).**
The scripts here only prepare its input and read its output; nothing in this repo calls a
model.

## Requirements

| | Why | Where |
|---|---|---|
| `vllm` | Runs the two LLM stages | A Linux GPU box — it has no macOS wheels. Run via `uvx`, no install; the wheel must match the driver's CUDA — see [running `vllm`](#running-vllm) |
| `jq` | Builds the batch request files | Anywhere |
| This repo, `uv sync`'d | Trains and evaluates the detector | Anywhere |
| `questions.json` | Your QA pack — see below | You write this |

The work falls into three phases, and only the middle one needs a GPU:

| Phase | What happens | GPU | Cost |
|---|---|---|---|
| **A. Prepare the data** | Write the questions, render them as batch requests | no | minutes |
| **B. Run the models** | Generate answers, then grade them — two `vllm run-batch` passes | **yes** | hours |
| **C. Train and evaluate** | Fit the detector on the labels, score it on held-out data | no | seconds |

So the usual split is to do A on a laptop, copy the request files to a GPU box for B, copy
the outputs back, and do C anywhere. Phase C is cheap enough to rerun freely — retraining
at a different `k` or on a different reduction never costs GPU time, because it reads the
same two files phase B produced.

## The training data

The detector is a logistic regression trained on `(response, was_it_a_hallucination)` pairs.
You supply the questions and their gold answers; the pipeline produces the responses, and
the LLM judge produces the labels.

The only file authored by hand is `questions.json`, a list of:

```json
[{"question": "Who sent Augustine to England?",
  "question_id": "q-1",
  "short_answer": "Pope Gregory",
  "answer_aliases": ["Gregory I", "the Pope"]}]
```

| Field | Required | Used for |
|---|---|---|
| `question` | yes | The prompt sent to the model being scored |
| `question_id` | yes | Becomes `custom_id`; every stage joins on it, so it must be unique |
| `short_answer` | yes | The gold answer the judge grades against |
| `answer_aliases` | no | Other answers the judge should accept; omit or leave empty |

The paper uses **TriviaQA** for training and **WebQuestions** to test generalisation, plus
a financial RAG corpus (ArGiMi-Ardian) for missing-context detection. Any short-form QA set
works, including domain-specific question sets. Two properties matter:

- **Answers must be short enough to grade automatically.** The judge compares against
  `short_answer`; an essay cannot be scored this way.
- **The set must be hard enough that the model gets some wrong.** The fit needs both
  classes. A model that answers everything correctly produces no hallucinations to learn
  from, and the fit will fail with a single-class error.

A few hundred questions is a workable start. Step 6 reports a 95% interval; if it is too
wide to tell EPR and WEPR apart, that is the signal to label more.

## Tutorial

Seven steps in three phases. Set these once — `K` has to be the same number in steps 2 and
6, and mismatching it is the most common way to train a detector that scores wrongly:

```bash
cd scripts/ecir
MODEL=mistralai/Ministral-8B-Instruct-2410   # the model being scored
JUDGE=mistralai/Ministral-8B-Instruct-2410   # the model that grades its answers
K=15                                          # ranks per token; every shipped file uses 15
OUT=out
mkdir -p "$OUT"
```

---

## Phase A — Prepare the data

No GPU. Everything here is cheap and worth getting right before spending GPU hours.

### Step 1 — write `questions.json`

The only hand-authored file. See [the training data](#the-training-data) above for the
schema and for the properties of a usable question set.

### Step 2 — build the generation requests

```bash
./build_generation_requests.sh questions.json "$MODEL" "$K" > "$OUT/gen_requests.jsonl"
```

One JSON line per question, in the OpenAI batch format, asking for `top_logprobs: $K`.
Check it before spending GPU time:

```bash
head -1 "$OUT/gen_requests.jsonl" | jq '{custom_id, model: .body.model, k: .body.top_logprobs}'
wc -l < "$OUT/gen_requests.jsonl"   # should equal the question count
```

Sampling follows the paper (§4.1.2): non-greedy decoding at `T_samp = 1.0`, `top_p = 1.0`,
sampling cutoff `K_samp = 50`. Override with `GEN_TEMPERATURE`, `GEN_TOP_P`, `GEN_TOP_K`.

The 200-token cap is this script's default, not the paper's — the paper only notes that the
tasks yield short answers. The paper does report that dropping to `T_samp = 0.6` changed
ROC-AUC by less than 1 point on Falcon-3-10B, so the signal is not an artefact of sampling
hot.

---

## Phase B — Run the models *(GPU)*

The expensive phase: two `vllm run-batch` passes, and they are **strictly sequential** —
the judge grades answers that do not exist until generation has finished.

### Running `vllm`

`vllm` is not a dependency of this repo and does not need installing — `uvx` fetches it for
the duration of the call:

```bash
uvx --python 3.12 vllm run-batch ...
```

**Match the wheel to the driver.** Each wheel is built against one CUDA major version. If
the driver is older than the wheel's CUDA, the engine aborts at startup:

```
RuntimeError: The NVIDIA driver on your system is too old (found version <NNNNN>)
```

or, depending on the pairing, `libcudart.so.<N>: cannot open shared object file`. The
default wheel on PyPI tracks the newest CUDA, so a box a generation behind needs a pinned
one instead.

Read the driver's CUDA version from `nvidia-smi`, then take the matching build from the
[vllm releases](https://github.com/vllm-project/vllm/releases) — each release publishes
several alongside the PyPI default — and point the torch index at the same one:

```bash
CUDA_TAG=cu129          # the build matching your driver
VLLM_VERSION=0.26.0
VLLM_WHEEL=https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_TAG}-cp38-abi3-manylinux_2_28_x86_64.whl

uvx --python 3.12 --index-strategy unsafe-best-match \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    --from "vllm @ $VLLM_WHEEL" \
    vllm run-batch ...
```

`--extra-index-url` matters as much as the wheel: without a matching torch build, the
resolver installs the default-CUDA torch and the same abort follows from there instead.

This guide was last run end-to-end against a driver reporting **CUDA 12.8**, using the
`cu129` build shown above under CUDA minor-version compatibility. A driver new enough for
the default wheel needs none of this — plain `uvx vllm run-batch` is enough.

On a shared box, let the scheduler hand out GPUs rather than pinning `CUDA_VISIBLE_DEVICES`
by hand, and cap the context if the model's default exceeds what one card holds
(`--max-model-len 4096 --gpu-memory-utilization 0.90`).

### Step 3 — generate the answers

```bash
vllm run-batch -i "$OUT/gen_requests.jsonl" -o "$OUT/responses.jsonl" --model "$MODEL"
```

Confirm every request came back with the right rank width:

```bash
jq -r 'select(.response != null)
       | (.response.body // .response).choices[0].logprobs.content[0].top_logprobs | length' "$OUT/responses.jsonl" | sort -u
```

One number should print, and it must equal `$K`. If it is smaller, the generation ignored
`top_logprobs` — fix that and rerun, because step 6 will refuse the data.

### Step 4 — build the judge requests

```bash
./build_judge_requests.sh questions.json "$OUT/responses.jsonl" "$JUDGE" > "$OUT/judge_requests.jsonl"
```

Joins each generated answer back to its gold answer on `custom_id` and renders the paper's
grading prompt. Generations that failed are dropped, and the count is reported on stderr.

### Step 5 — grade the answers

```bash
vllm run-batch -i "$OUT/judge_requests.jsonl" -o "$OUT/judgments.jsonl" --model "$JUDGE"
```

The judge replies with `{"judgment": true|false, "explanation": "..."}`. `true` means the
answer was **correct**, so the training label is its negation — 1 marks a hallucination.
Spot-check a few, then check the class balance:

```bash
jq -r 'select(.response != null) | (.response.body // .response).choices[0].message.content' "$OUT/judgments.jsonl" | head -3

jq -r 'select(.response != null) | (.response.body // .response).choices[0].message.content' "$OUT/judgments.jsonl" \
  | grep -c '"judgment": *true'
```

Compare that count against the question count. All-correct or all-wrong cannot be fit;
step 1 then needs harder or easier questions.

---

## Phase C — Train and evaluate

No GPU, seconds to run, and it never needs phase B repeated.

### Step 6 — train the detector and evaluate it

```bash
uv run ../train_detector.py \
    --responses "$OUT/responses.jsonl" \
    --judgments "$OUT/judgments.jsonl" \
    --reduction wepr --k "$K" \
    --output "$OUT/wepr.skops" \
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
wrote out/wepr.skops

               precision    recall  f1-score   support

    grounded       0.88      0.93      0.90        72
hallucination      0.74      0.61      0.67        28

    accuracy                           0.84       100

   roc_auc: 0.7412  [0.6810, 0.7955]
    pr_auc: 0.5233  [0.4401, 0.6118]
```

The two halves answer different questions. ROC-AUC and PR-AUC score the *ranking*, which
governs triage by score; the classification report scores the decisions at the 0.5
threshold. A detector can rank well and still decide poorly there, so both are relevant,
and only the AUCs carry over to a different threshold. Recall on the `hallucination` row is
usually the figure of interest: the fraction of hallucinations actually flagged.

`--test_size 0` fits on everything and skips the evaluation. It is appropriate only once
the procedure has been measured and the remaining data is wanted in the model.

**The two reductions are independent and can run at the same time** — both read the same
two files, and neither needs phase B repeated:

```bash
for reduction in epr wepr; do
  uv run ../train_detector.py \
      --responses "$OUT/responses.jsonl" --judgments "$OUT/judgments.jsonl" \
      --reduction "$reduction" --k "$K" \
      --output "$OUT/${reduction}.skops" \
      --report "$OUT/${reduction}_evaluation.json" &
done
wait
```

Sweeping `--k` works the same way, with one constraint: `--k` can go no higher than the
`top_logprobs` phase B was run with. Generating wide allows retraining narrower at no
cost; generating narrow can only be corrected with more GPU time.

### Step 7 — use the trained detector

```python
from artefactual.scoring import wepr

detector = wepr("out/wepr.skops", k=15)
detector.predict_proba(response)[:, 1]
```

The file is the same `.skops` format the published detectors use, so it is interchangeable
with them. Score responses generated the same way — same model, and `top_logprobs` at least `k`.

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
it was not trained on. Training data should resemble the traffic being scored.

SelfCheckGPT needs 10 extra generations per answer for those numbers; EPR and WEPR need
none. The paper measures roughly 80 ± 20 µs per score against at least 10 s for
SelfCheckGPT.

## Use k = 15

`k` is the number of ranks per token, and it appears in two places that **must agree**:
`build_generation_requests.sh` (step 2, where it becomes `top_logprobs`) and `--k` on
`train_detector.py` (step 6). Every shipped detector was trained at 15. Setting `K` once at
the top of the tutorial is what keeps them in step.

It cannot be inferred, because it is part of the metric: EPR is the entropy of the top `k`
ranks (Eq. 3 and 6 of the paper), so changing `k` changes what is being measured, and WEPR
has one coefficient per rank.

Generating with a smaller `top_logprobs` than the fitted `k` fails loudly, for both
reductions, as soon as the responses are read:

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
at the rank count they were trained at; pass k=15, or supply weights trained at k=20.
```

Responses generated at a narrower `k` must be regenerated; a detector trained at another
rank count is not a substitute.
## If something looks wrong

**`Response N carries M rank(s) per token but k=15 was requested`.** The generation batch
was produced with a smaller `top_logprobs` than the fitted `k`, most often because step 2
and step 6 were run with different values. The responses can be inspected directly:

```bash
jq -r 'select(.response != null)
       | (.response.body // .response).choices[0].logprobs.content[0].top_logprobs | length' out/responses.jsonl | sort -u
```

One number should come back, and it must be at least `--k`. If it is smaller, rerun
steps 2 and 3 — the judgments are unaffected and do not need regenerating.

**`joined N pairs on custom_id` reports fewer than the question count.** Some requests
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

```text
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
prompts and the run no longer reproduces the paper.

The judge's `judgment: true` means the answer was correct, so the training label is its
negation: **1 marks a hallucination**.

## Evaluation method

`train_detector.py` splits the labelled set once with `train_test_split`, stratified so the
hallucination rate is the same on both sides, fits the detector on the training part, and
scores it on the held-out part. **The evaluation is of the model that was written to
`--output`** — the detector is fitted once and never refitted, so the numbers belong to the
artefact being kept rather than to the procedure that produced it.

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
0.5 threshold. The two measure different things: the AUCs score the *ranking*, which governs
triage by score, while the report scores the decisions. Recall on the `hallucination` row is
usually the figure of interest — the fraction actually flagged.

## Not reproduced

The SelfCheckGPT and HalluDetect baselines the paper compares against are not part of the
EPR/WEPR method and are not included here, so the paper's two comparison columns cannot be
rebuilt from this repo.
