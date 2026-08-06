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
#   GEN_TEMPERATURE  sampling temperature      (default 1.0, the paper's T_samp)
#   GEN_TOP_P        nucleus sampling cutoff   (default 1.0, the paper's setting)
#   GEN_TOP_K        sampling cutoff K_samp    (default 50, the paper's setting)
#   GEN_MAX_TOKENS   answer length cap         (default 200; the paper does not state
#                    one, it only notes the tasks yield short answers)
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

# Paper section 4.1.2: non-greedy decoding at T_samp = 1.0, top_p = 1.0, K_samp = 50.
# Non-greedy is the point -- the method measures hesitation in the raw distribution, and
# the paper reports the signal survives conservative decoding rather than requiring it.
temperature=${GEN_TEMPERATURE:-1.0}
top_p=${GEN_TOP_P:-1.0}
top_k=${GEN_TOP_K:-50}
max_tokens=${GEN_MAX_TOKENS:-200}

[ -f "$questions" ] || { echo "error: no such question pack: $questions" >&2; exit 1; }

jq -c \
  --rawfile tpl "$here/prompts/generate.txt" \
  --arg model "$model" \
  --argjson k "$k" \
  --argjson temperature "$temperature" \
  --argjson top_p "$top_p" \
  --argjson top_k "$top_k" \
  --argjson max_tokens "$max_tokens" '
  .[]
  # Bound before the template chain: inside join(), the input is the array split()
  # produced, not the question object.
  | .question as $question
  | ($tpl | split("{{query}}") | join($question)) as $prompt
  | {custom_id: .question_id, method: "POST", url: "/v1/chat/completions",
     body: {model: $model,
            logprobs: true, top_logprobs: $k,
            temperature: $temperature, top_p: $top_p, top_k: $top_k,
            max_completion_tokens: $max_tokens,
            messages: [{role: "user", content: $prompt}]}}
' "$questions"
