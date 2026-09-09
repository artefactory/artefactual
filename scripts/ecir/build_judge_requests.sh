#!/usr/bin/env bash
#
# Build `vllm run-batch` requests that grade generated answers against gold ones.
#
# Usage:
#   build_judge_requests.sh <questions.json> <responses.jsonl> <judge-model> > judge_requests.jsonl
#
# Arguments:
#   questions.json   the same pack used for generation, carrying the gold answers
#   responses.jsonl  run-batch output from the generation stage
#   judge-model      model to grade with, as vllm names it
#
# Environment:
#   JUDGE_TEMPERATURE  sampling temperature (default 0, grading should be deterministic)
#   JUDGE_MAX_TOKENS   reply cap            (default 200; raise it if verdicts come back
#                      truncated and train_detector.py reports unparsed judgments)
#
# Joins the generations back to their gold answers on `custom_id`, which
# `vllm run-batch` carries through from the generation request. Rows whose generation
# failed are dropped and counted on stderr; a request the server rejected reports that as
# a non-2xx `status_code` with `error` still null, so the status decides, not `error`.
# Lines are read as the OpenAI Batch output spec defines them -- no other shape is accepted.
#
# The prompt is rendered by literal split/join rather than regex substitution, so a
# question containing backslashes or `&` cannot corrupt it. `tests/test_ecir_prompts.py`
# checks the result byte-for-byte against the original Jinja template.
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit 0
fi

questions=${1:?usage: build_judge_requests.sh questions.json responses.jsonl judge-model}
responses=${2:?missing responses.jsonl}
model=${3:?missing judge model}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for file in "$questions" "$responses"; do
  [ -f "$file" ] || { echo "error: no such file: $file" >&2; exit 1; }
done

# The judge answers in JSON ({"judgment": ..., "explanation": ...}), so it needs room to
# reply -- this is not a single-token verdict.
temperature=${JUDGE_TEMPERATURE:-0}
max_tokens=${JUDGE_MAX_TOKENS:-200}

# A line carries a usable generation only when the request succeeded. The Batch spec says
# that two ways: top-level `error` for non-HTTP failures, and -- for a request the server
# rejects -- `error` null with a non-2xx `status_code` and an error object sitting where
# the completion would be. So a body is not evidence of a completion.
read -r -d '' usable <<'JQ' || true
def usable:
  if .error != null or .response == null then false
  else
    (.response.status_code
     // error("custom_id \(.custom_id): response envelope carries no status_code")) as $status
    | $status >= 200 and $status < 300 and .response.body != null
  end;
JQ

total=$(wc -l <"$responses" | tr -d " ")
kept=$(jq -s "$usable"' map(select(usable)) | length' "$responses")
if [ "$kept" -ne "$total" ]; then
  echo "warning: dropping $((total - kept))/$total generations that failed" >&2
fi

jq -c -s \
  --rawfile tpl "$here/prompts/judge.txt" \
  --slurpfile questions "$questions" \
  --arg model "$model" \
  --argjson temperature "$temperature" \
  --argjson max_tokens "$max_tokens" "$usable"'
  ($questions[0] | INDEX(.question_id)) as $gold
  | .[]
  | select(usable)
  | .custom_id as $id
  | ($gold[$id] // error("no question for custom_id \($id)")) as $q
  # The Batch output spec: .response is {status_code, request_id, body} and the completion
  # is the body.
  | .response.body as $completion
  # Bound before the template chain: inside join(), `.` is the array split() produced,
  # not the response object.
  | $completion.choices[0].message.content as $answer
  | ($q.answer_aliases // []) as $aliases
  # Jinja runs with trim_blocks off, so every {% %} tag leaves the newline it sits on.
  # That is why each alias is wrapped in newlines and the block keeps a trailing pair:
  # verified byte-for-byte against jinja2 in tests/test_ecir_prompts.py.
  | (if ($aliases | length) > 0
     then "\nAnswer Aliases (Additional Correct Answers):\n"
          + ($aliases | map("\n- \(.)\n") | add)
          + "\n\n"
     else "\n" end) as $aliases_block
  # jinja2 drops the final template newline: keep_trailing_newline defaults to false.
  | ($tpl
     | rtrimstr("\n")
     | split("{{query}}")            | join($q.question)
     | split("{{expected_answer}}")  | join($q.short_answer)
     | split("{{aliases_block}}")    | join($aliases_block)
     | split("{{generated_answer}}") | join($answer)
    ) as $prompt
  | {custom_id: $id, method: "POST", url: "/v1/chat/completions",
     body: {model: $model, temperature: $temperature, max_completion_tokens: $max_tokens,
            messages: [{role: "user", content: $prompt}]}}
' "$responses"
