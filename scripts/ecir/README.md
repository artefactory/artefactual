# ECIR reproduction: EPR / WEPR calibration

Reproduces the calibration pipeline for the EPR and WEPR detectors from
*"Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate"*.

Generation and judging are plain [`vllm run-batch`](https://docs.vllm.ai/en/stable/examples/offline_inference/openai_batch.html)
invocations rather than bespoke scripts. Only the calibration fit needs this package.

```bash
./run_pipeline.sh questions.json out/ <generator-model> <judge-model> 15
# -> out/epr_weights.json, out/wepr_weights.json
```

`questions.json` is a list of `{question, question_id, short_answer, answer_aliases}`.

## Stages

| Stage | Command | Output |
|---|---|---|
| 1 | `build_generation_requests.sh` | `gen_requests.jsonl` |
| 2 | `vllm run-batch` | `responses.jsonl` — generations with `top_logprobs` |
| 3 | `build_judge_requests.sh` | `judge_requests.jsonl` |
| 4 | `vllm run-batch` | `judgments.jsonl` — `{"judgment": …}` verdicts |
| 5 | `../train_calibration.py` | `{epr,wepr}_weights.json` |

`question_id` becomes the batch `custom_id`, which `run-batch` carries into its output.
Every stage joins on it, so a reordered or partially failed batch cannot pair a generation
with the wrong verdict. Rows whose request failed (`error != null`) are dropped and counted.

## `k` is not a free parameter

EPR is a *mean* over ranks, so the rank count is part of the feature definition, and WEPR
coefficients are fixed at the rank count they were fit on. The `k` passed to
`build_generation_requests.sh` (as `top_logprobs`) must match the `--k` given to
`train_calibration.py`. All shipped calibrations use `k = 15`.

## Prompts

`prompts/generate.txt` and `prompts/judge.txt` hold the paper's prompts verbatim, with
`{{placeholders}}` substituted by `jq` using literal split/join — so a question containing
backslashes, `&` or quotes cannot corrupt the rendering.

`prompts/judge.jinja` is the original jinja2 template, kept so the `jq` rendering can be
re-checked against it. It was verified byte-identical for 0, 1 and 2 aliases; jinja runs
with `trim_blocks` off, so every `{% %}` tag leaves the newline it sits on and the
template's trailing newline is dropped — that whitespace is reproduced exactly.

## Not reproduced

The SelfCheckGPT and Halludetect baselines are not part of the EPR/WEPR method and are not
included. The paper's bootstrap evaluation harness (1000-repetition resampling reporting
ROC-AUC / PR-AUC) is not here either — this pipeline fits calibrations, it does not
evaluate them.
