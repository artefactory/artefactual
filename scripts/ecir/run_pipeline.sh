#!/usr/bin/env bash
#
# Build an EPR/WEPR calibration from a question pack, end to end.
#
# Usage:
#   run_pipeline.sh [options] <questions.json> <outdir>
#
# Options (each has an environment-variable equivalent, shown in brackets):
#   -m, --model MODEL       model that answers the questions   [GEN_MODEL]
#   -j, --judge MODEL       model that grades the answers      [JUDGE_MODEL]
#   -k, --top-logprobs K    ranks per token to request         [TOP_LOGPROBS]
#   -r, --reductions LIST   space-separated: "epr wepr"        [REDUCTIONS]
#   -f, --force             redo stages whose output exists
#   -h, --help              show this help
#
# Further knobs, environment only:
#   GEN_TEMPERATURE    sampling temperature for the answers      (default 0.6)
#   GEN_MAX_TOKENS     answer length cap                         (default 200)
#   JUDGE_TEMPERATURE  sampling temperature for the judge        (default 0)
#   JUDGE_MAX_TOKENS   judge reply cap, raise it if verdicts     (default 200)
#                      come back truncated
#   PYTHON             interpreter that can import artefactual   (default python3)
#
# The two LLM stages are plain `vllm run-batch` calls; the rest is this package. Every
# stage keys on `custom_id`, which run-batch carries from request to response, so stages
# join by id rather than by line order.
#
# Stages are skipped when their output file already exists and is non-empty, so an
# interrupted run resumes and a re-fit does not regenerate. Pass --force to override.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; }

gen_model=${GEN_MODEL:-mistralai/Ministral-8B-Instruct-2410}
judge_model=${JUDGE_MODEL:-mistralai/Ministral-8B-Instruct-2410}
k=${TOP_LOGPROBS:-15}
reductions=${REDUCTIONS:-"epr wepr"}
force=0
positional=()

while [ $# -gt 0 ]; do
  case $1 in
    -m | --model) gen_model=${2:?--model needs a value}; shift 2 ;;
    -j | --judge) judge_model=${2:?--judge needs a value}; shift 2 ;;
    -k | --top-logprobs) k=${2:?--top-logprobs needs a value}; shift 2 ;;
    -r | --reductions) reductions=${2:?--reductions needs a value}; shift 2 ;;
    -f | --force) force=1; shift ;;
    -h | --help) usage; exit 0 ;;
    --) shift; positional+=("$@"); break ;;
    -*) echo "error: unknown option $1 (try --help)" >&2; exit 2 ;;
    *) positional+=("$1"); shift ;;
  esac
done

if [ ${#positional[@]} -ne 2 ]; then
  usage >&2
  exit 2
fi
questions=${positional[0]}
outdir=${positional[1]}

python_bin=${PYTHON:-python3}

for tool in jq vllm "$python_bin"; do
  command -v "$tool" >/dev/null || { echo "error: $tool is not on PATH" >&2; exit 1; }
done

# Checked up front rather than at stage 5: the two GPU stages run first and are by far the
# expensive ones, so an unusable interpreter must not be discovered after them. The symbols
# the fit needs are imported, not just the package -- another checkout on sys.path can
# satisfy `import artefactual` and still be too old to satisfy the scripts.
if ! "$python_bin" -c 'from artefactual.scoring import epr, wepr' >/dev/null 2>&1; then
  echo "error: $python_bin cannot import epr/wepr from artefactual." >&2
  "$python_bin" -c 'import artefactual; print("       it resolved artefactual to:", artefactual.__file__)' 2>/dev/null || true
  echo "       Activate the project environment, or set PYTHON, e.g.:" >&2
  echo "         PYTHON=\$(uv python find) $(basename "${BASH_SOURCE[0]}") ..." >&2
  echo "         PYTHON=./.venv/bin/python  $(basename "${BASH_SOURCE[0]}") ..." >&2
  exit 1
fi
[ -f "$questions" ] || { echo "error: no such question pack: $questions" >&2; exit 1; }
mkdir -p "$outdir"

# A stage is done when its output exists and is non-empty; --force ignores that.
done_already() {
  [ "$force" -eq 0 ] && [ -s "$1" ]
}

echo "==> config"
echo "    questions : $questions"
echo "    outdir    : $outdir"
echo "    generator : $gen_model"
echo "    judge     : $judge_model"
echo "    k         : $k"
echo "    reductions: $reductions"
echo "    python    : $python_bin"
echo "    artefactual: $("$python_bin" -c 'import artefactual, pathlib; print(pathlib.Path(artefactual.__file__).parent)')"

echo "==> 1/6 building generation requests"
"$here/build_generation_requests.sh" "$questions" "$gen_model" "$k" >"$outdir/gen_requests.jsonl"

echo "==> 2/6 generating answers (GPU)"
if done_already "$outdir/responses.jsonl"; then
  echo "    skipped: $outdir/responses.jsonl exists (--force to redo)"
else
  vllm run-batch -i "$outdir/gen_requests.jsonl" -o "$outdir/responses.jsonl" --model "$gen_model"
fi

echo "==> 3/6 building judge requests"
"$here/build_judge_requests.sh" "$questions" "$outdir/responses.jsonl" "$judge_model" \
  >"$outdir/judge_requests.jsonl"

echo "==> 4/6 grading answers (GPU)"
if done_already "$outdir/judgments.jsonl"; then
  echo "    skipped: $outdir/judgments.jsonl exists (--force to redo)"
else
  vllm run-batch -i "$outdir/judge_requests.jsonl" -o "$outdir/judgments.jsonl" --model "$judge_model"
fi

echo "==> 5/6 fitting calibrations"
for reduction in $reductions; do
  "$python_bin" "$repo/scripts/train_calibration.py" \
    --responses "$outdir/responses.jsonl" \
    --judgments "$outdir/judgments.jsonl" \
    --reduction "$reduction" \
    --k "$k" \
    --output "$outdir/${reduction}_weights.json"
done

echo "==> 6/6 evaluating (out-of-bag bootstrap)"
for reduction in $reductions; do
  "$python_bin" "$repo/scripts/evaluate_calibration.py" \
    --responses "$outdir/responses.jsonl" \
    --judgments "$outdir/judgments.jsonl" \
    --reduction "$reduction" \
    --k "$k" \
    --output "$outdir/${reduction}_evaluation.json"
done

echo "done. calibrations in $outdir:"
for reduction in $reductions; do
  echo "    ${reduction}_weights.json  ${reduction}_evaluation.json"
done
