#!/usr/bin/env bash
#
# Build a question pack from a Hugging Face QA dataset.
#
# Usage:
#   build_questions.sh <triviaqa|webquestions> [n] > questions.json
#
# Arguments:
#   dataset  triviaqa     the paper's training set, sampled
#            webquestions the paper's generalisation set, taken whole
#   n        questions to sample, triviaqa only (default 500)
#
# Environment:
#   QUESTIONS_SEED  shuffle seed for the triviaqa sample (default 42)
#
# Writes the schema steps 2 and 4 read: a JSON list of
# {question, question_id, short_answer, answer_aliases}. A pack can equally be written by
# hand in that schema -- the rest of the pipeline cannot tell the two apart.
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

dataset=${1:?usage: build_questions.sh triviaqa|webquestions [n]}
n=${2:-500}
seed=${QUESTIONS_SEED:-42}

rows=$(mktemp)
trap 'rm -f "$rows"' EXIT

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

case "$dataset" in
  triviaqa)     fetch=$fetch_triviaqa;     shape=$shape_triviaqa ;;
  webquestions) fetch=$fetch_webquestions; shape=$shape_webquestions ;;
  *) echo "error: unknown dataset: $dataset (expected triviaqa or webquestions)" >&2; exit 1 ;;
esac

uv run --no-project --with datasets python -c "$fetch" "$seed" "$n" "$rows" >&2
jq -s "$shape" "$rows"
