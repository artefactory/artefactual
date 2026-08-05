#!/usr/bin/env bash
# Build LLM-as-a-judge batch requests from generated answers.
#
#   build_judge_requests.sh questions.json responses.jsonl <judge-model> > judge_requests.jsonl
#
# Joins the generations back to their gold answers on `custom_id`, which
# `vllm run-batch` carries through from the generation request. Rows where
# generation failed (`error != null`) are dropped and counted on stderr.
#
# The prompt is rendered by literal split/join rather than regex substitution, so a
# question containing backslashes or `&` cannot corrupt it. `tests/test_ecir_prompts.py`
# checks the result byte-for-byte against the original Jinja template.
set -euo pipefail

questions=${1:?usage: build_judge_requests.sh questions.json responses.jsonl judge-model}
responses=${2:?missing responses.jsonl}
model=${3:?missing judge model}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# The judge answers in JSON ({"judgment": ..., "explanation": ...}), so it needs room to
# reply -- this is not a single-token verdict.
max_tokens=${JUDGE_MAX_TOKENS:-200}

total=$(wc -l <"$responses" | tr -d " ")
kept=$(jq -s 'map(select(.error == null and .response != null)) | length' "$responses")
if [ "$kept" -ne "$total" ]; then
  echo "warning: dropping $((total - kept))/$total generations that failed" >&2
fi

jq -c -s \
  --rawfile tpl "$here/prompts/judge.txt" \
  --slurpfile questions "$questions" \
  --arg model "$model" \
  --argjson max_tokens "$max_tokens" '
  ($questions[0] | INDEX(.question_id)) as $gold
  | .[]
  | select(.error == null and .response != null)
  | .custom_id as $id
  | ($gold[$id] // error("no question for custom_id \($id)")) as $q
  # Bound before the template chain: inside join(), `.` is the array split() produced,
  # not the response object.
  | .response.choices[0].message.content as $answer
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
     body: {model: $model, temperature: 0, max_completion_tokens: $max_tokens,
            messages: [{role: "user", content: $prompt}]}}
' "$responses"
