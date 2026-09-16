#!/usr/bin/env bash
#
# Build a question pack from a Hugging Face QA dataset.
#
# Usage:
#   build_questions.sh <triviaqa|simpleqa|webquestions|mixed> [n] > questions.json
#
# Arguments:
#   dataset  triviaqa     the paper's training set, sampled
#            simpleqa     short fact-seeking questions selected to be hard
#            webquestions the paper's generalisation set, taken whole
#            mixed        half triviaqa, half simpleqa -- the default pack
#   n        questions to sample (default 500; 100 for mixed, split evenly).
#            Ignored for webquestions, which is taken whole.
#
# Environment:
#   QUESTIONS_SEED  shuffle seed for every sampled set (default 42)
#
# Writes the schema steps 2 and 4 read: a JSON list of
# {question, question_id, short_answer, answer_aliases}. A pack can equally be written by
# hand in that schema -- the rest of the pipeline cannot tell the two apart.
#
# `mixed` exists because TriviaQA alone is too easy. A current endpoint answers most of it
# correctly, so `y = int(not verdict)` is nearly all zeros and the fit sees a separable
# problem; the ROC-AUC that comes out then describes the question set. SimpleQA was built
# to be hard, so mixing the two puts real mass in both classes.
#
# `uv` fetches `datasets` into an ephemeral environment. `--no-project` keeps that
# independent of the checkout it runs from: building the repo is not needed to shape a
# question pack, and an export without version metadata cannot be built at all. Rows land
# in a temporary file and are deleted on exit.
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit 0
fi

dataset=${1:?usage: build_questions.sh triviaqa|simpleqa|webquestions|mixed [n]}
seed=${QUESTIONS_SEED:-42}

# TriviaQA's closed-book configuration carries every field the pack needs. The split
# arrives grouped by source, so shuffling before sampling is what makes `n` rows a sample
# of the set rather than of its first source.
read -r -d '' fetch_triviaqa <<'PY' || true
import sys
from datasets import load_dataset
split = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
split.shuffle(seed=int(sys.argv[1])).select(range(int(sys.argv[2]))).to_json(sys.argv[3])
PY

# `rc.nocontext` is `rc.web.nocontext` and `rc.wikipedia.nocontext` concatenated (9,951 +
# 7,993 = 17,944 validation rows, over at most 11,313 distinct questions), so a question
# with both kinds of evidence appears twice under the same id. What told the two rows apart
# is the evidence, which is exactly what `.nocontext` drops, so the fields read here are the
# question's own annotation either way -- and keeping both would pay for the same
# generation twice. Expect a little under `n` questions out of `n` rows.
#
# TriviaQA questions carry a median of 8 aliases and a long tail -- mean 13, and 158 at the
# widest, measured over a 500-row sample -- every one of them rendered into the judge
# prompt. `unique_by(ascii_downcase)` collapses the ones that differ only in case; the
# `select` drops the ones that restate the gold answer, which the prompt already carries on
# its own line.
read -r -d '' shape_triviaqa <<'JQ' || true
[ .[]
  # Bound here because inside the `select` below `.` is the alias, not the row.
  | .answer.value as $gold
  | {question,
     question_id,
     short_answer: $gold,
     answer_aliases: ([.answer.aliases[] | select(ascii_downcase != ($gold | ascii_downcase))]
                      | unique_by(ascii_downcase))} ]
| unique_by(.question_id)
JQ

# SimpleQA is one CSV with no configs and no id column, so the row's position in the full
# test split is its id -- the same reasoning WebQuestions gets below. `row_index` is added
# before the shuffle for exactly that reason: taken after, the id would name a position in
# this sample, and a new seed or size would rebind it while a stale responses.jsonl still
# carried the old one.
read -r -d '' fetch_simpleqa <<'PY' || true
import sys
from datasets import load_dataset
split = load_dataset("basicv8vc/SimpleQA", split="test")
split = split.add_column("row_index", list(range(len(split))))
split.shuffle(seed=int(sys.argv[1])).select(range(int(sys.argv[2]))).to_json(sys.argv[3])
PY

# SimpleQA is graded against a single canonical answer by design: the benchmark's own
# criterion is that the answer be unambiguous, so there is nothing to put in
# `answer_aliases` and an empty list is the honest value rather than a gap.
# `build_judge_requests.sh` already renders an empty list as no alias block at all.
# The `sq-` prefix keeps these out of TriviaQA's `tc_*` and WebQuestions' `wq-*` namespaces.
read -r -d '' shape_simpleqa <<'JQ' || true
[ .[]
  | select((.answer // "") != "" and (.problem // "") != "")
  | {question: .problem,
     question_id: "sq-\(.row_index)",
     short_answer: .answer,
     answer_aliases: []} ]
JQ

# WebQuestions has no config and no id column, and carries one flat answer list that serves
# both answer fields. Its test split is 2,032 questions, small enough to take whole -- which
# is also what makes the row position a usable id, since it is the same number on every run.
# Sample first and the id would mean a position in that sample, so a new seed or size
# rebinds it and a stale responses.jsonl would join old generations to new questions with
# nothing to report it. The `wq-` prefix keeps these out of a hand-written `q-1` namespace.
read -r -d '' fetch_webquestions <<'PY' || true
import sys
from datasets import load_dataset
load_dataset("stanfordnlp/web_questions", split="test").to_json(sys.argv[3])
PY

# jq has no error for reading a field that is not there: `.answers[0]` on an empty list is
# null, and a null gold answer reaches the judge as an empty string, which grades every
# answer against nothing. Dropped here instead. The split has no such row today; the guard
# is against pointing this at a set that does.
read -r -d '' shape_webquestions <<'JQ' || true
[ to_entries[]
  | select((.value.answers | length) > 0)
  | .value.answers[0] as $gold
  | {question: .value.question,
     question_id: "wq-\(.key)",
     short_answer: $gold,
     answer_aliases: ([.value.answers[1:][] | select(ascii_downcase != ($gold | ascii_downcase))]
                      | unique_by(ascii_downcase))} ]
JQ

# One source, shaped into the pack schema and written to stdout. Each call gets its own
# temporary file so `mixed` can run two without them colliding; the trap is set per call
# rather than once, because a single EXIT trap cannot name a file that does not exist yet.
pack() {
  local name=$1 count=$2 fetch shape rows
  case "$name" in
    triviaqa)     fetch=$fetch_triviaqa;     shape=$shape_triviaqa ;;
    simpleqa)     fetch=$fetch_simpleqa;     shape=$shape_simpleqa ;;
    webquestions) fetch=$fetch_webquestions; shape=$shape_webquestions ;;
    *) echo "error: unknown dataset: $name" >&2; return 1 ;;
  esac

  rows=$(mktemp)
  # shellcheck disable=SC2064  # $rows must expand now, not when the trap fires.
  trap "rm -f '$rows'" RETURN

  uv run --no-project --with datasets python -c "$fetch" "$seed" "$count" "$rows" >&2
  jq -s "$shape" "$rows"
}

case "$dataset" in
  triviaqa|simpleqa|webquestions)
    pack "$dataset" "${2:-500}"
    ;;
  mixed)
    n=${2:-100}
    # An odd `n` would silently give one source the extra question. Refused rather than
    # rounded: the point of the mix is that neither half is the majority class.
    if [ $((n % 2)) -ne 0 ]; then
      echo "error: mixed needs an even n (got $n)" >&2
      exit 1
    fi
    half=$((n / 2))
    # Both halves are shuffled with the same seed against different sets, so the pack is
    # reproducible from `QUESTIONS_SEED` alone. Concatenated rather than interleaved: the
    # order a pack is read in never reaches the detector, and `unique_by` across two id
    # namespaces would be a no-op.
    jq -s 'add' <(pack triviaqa "$half") <(pack simpleqa "$half")
    ;;
  *)
    echo "error: unknown dataset: $dataset (expected triviaqa, simpleqa, webquestions or mixed)" >&2
    exit 1
    ;;
esac
