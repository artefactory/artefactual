#!/usr/bin/env bash
# Reproduce the ECIR EPR/WEPR calibration end to end.
#
#   run_pipeline.sh questions.json outdir [generator-model] [judge-model] [k]
#
# Generation and judging are plain `vllm run-batch` invocations; only the calibration
# fit and its evaluation need this package. Every stage keys on `custom_id`, which
# run-batch carries from the request into its output, so the stages join by id rather
# than by line order.
set -euo pipefail

questions=${1:?usage: run_pipeline.sh questions.json outdir [gen-model] [judge-model] [k]}
outdir=${2:?missing output directory}
gen_model=${3:-mistralai/Ministral-8B-Instruct-2410}
judge_model=${4:-mistralai/Ministral-8B-Instruct-2410}
k=${5:-15}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)

for tool in jq vllm; do
  command -v "$tool" >/dev/null || { echo "error: $tool is not on PATH" >&2; exit 1; }
done
mkdir -p "$outdir"

echo "==> 1/6 building generation requests (k=$k)"
"$here/build_generation_requests.sh" "$questions" "$gen_model" "$k" >"$outdir/gen_requests.jsonl"

echo "==> 2/6 generating"
vllm run-batch -i "$outdir/gen_requests.jsonl" -o "$outdir/responses.jsonl" --model "$gen_model"

echo "==> 3/6 building judge requests"
"$here/build_judge_requests.sh" "$questions" "$outdir/responses.jsonl" "$judge_model" \
  >"$outdir/judge_requests.jsonl"

echo "==> 4/6 judging"
vllm run-batch -i "$outdir/judge_requests.jsonl" -o "$outdir/judgments.jsonl" --model "$judge_model"

echo "==> 5/6 fitting calibrations"
for reduction in epr wepr; do
  python "$repo/scripts/train_calibration.py" \
    --responses "$outdir/responses.jsonl" \
    --judgments "$outdir/judgments.jsonl" \
    --reduction "$reduction" \
    --k "$k" \
    --output "$outdir/${reduction}_weights.json"
done

echo "==> 6/6 evaluating (out-of-bag bootstrap)"
for reduction in epr wepr; do
  python "$repo/scripts/evaluate_calibration.py" \
    --responses "$outdir/responses.jsonl" \
    --judgments "$outdir/judgments.jsonl" \
    --reduction "$reduction" \
    --k "$k" \
    --output "$outdir/${reduction}_evaluation.json"
done

echo "done: $outdir/{epr,wepr}_{weights,evaluation}.json"
