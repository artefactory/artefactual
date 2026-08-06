#!/usr/bin/env bash
#
# Build `vllm run-batch` requests that answer a question pack, with logprobs.
#
# Usage:
#   build_generation_requests.sh <questions.json> <model> [k] > requests.jsonl
#
# Arguments:
#   questions.json  list of {question, question_id, short_answer, answer_aliases}
#   model           model to answer with, as vllm names it
#   k               top logprobs per token (default 15)
#
# Environment:
#   GEN_TEMPERATURE  sampling temperature (default 0.6, the paper's setting)
#   GEN_MAX_TOKENS   answer length cap    (default 200, the paper's setting)
#
# `question_id` becomes `custom_id`, which run-batch carries into its output so every
# later stage joins by id rather than by line order.
#
# `k` must match the `--k` passed when fitting: EPR averages over exactly k ranks, so the
# rank count is part of the feature definition, and the fit refuses narrower responses.
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit 0
fi

questions=${1:?usage: build_generation_requests.sh questions.json model [k]}
model=${2:?missing model}
k=${3:-15}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

temperature=${GEN_TEMPERATURE:-0.6}
max_tokens=${GEN_MAX_TOKENS:-200}

[ -f "$questions" ] || { echo "error: no such question pack: $questions" >&2; exit 1; }

jq -c \
  --rawfile tpl "$here/prompts/generate.txt" \
  --arg model "$model" \
  --argjson k "$k" \
  --argjson temperature "$temperature" \
  --argjson max_tokens "$max_tokens" '
  .[]
  # Bound before the template chain: inside join(), the input is the array split()
  # produced, not the question object.
  | .question as $question
  | ($tpl | split("{{query}}") | join($question)) as $prompt
  | {custom_id: .question_id, method: "POST", url: "/v1/chat/completions",
     body: {model: $model,
            logprobs: true, top_logprobs: $k,
            temperature: $temperature, max_completion_tokens: $max_tokens,
            messages: [{role: "user", content: $prompt}]}}
' "$questions"
