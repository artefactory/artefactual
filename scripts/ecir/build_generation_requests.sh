#!/usr/bin/env bash
# Build generation batch requests from a question pack.
#
#   build_generation_requests.sh questions.json <model> [k] > requests.jsonl
#
# `questions.json` is a list of {question, question_id, short_answer, answer_aliases}.
# `question_id` becomes `custom_id`, which vllm run-batch carries into the output and the
# later stages join on.
#
# k is the number of top logprobs per token. It must match the k passed to the detector:
# EPR is a mean over ranks, so the rank count is part of the feature definition.
set -euo pipefail

questions=${1:?usage: build_generation_requests.sh questions.json model [k]}
model=${2:?missing model}
k=${3:-15}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Paper settings: single user turn, temperature 0.6, 200 new tokens.
temperature=${GEN_TEMPERATURE:-0.6}
max_tokens=${GEN_MAX_TOKENS:-200}

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
