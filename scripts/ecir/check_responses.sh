#!/usr/bin/env bash
#
# Report what a `vllm run-batch` output file actually contains.
#
# Usage:
#   check_responses.sh <batch_output.jsonl>
#
# Prints, in order: how many lines carry a usable completion, the custom_id and status of
# every line that does not, and the distinct rank widths present. Exits non-zero when
# nothing is usable.
#
# The rank width is the number this is worth running for. `top_logprobs` is a request the
# server may ignore, and a batch generated at a narrower width than the fit expects is only
# discovered by step 6 -- two GPU stages later.
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit 0
fi

responses=${1:?usage: check_responses.sh batch_output.jsonl}
[ -f "$responses" ] || { echo "error: no such file: $responses" >&2; exit 1; }

# The same rule build_judge_requests.sh applies. The Batch spec reports failure two ways --
# top-level `error` for non-HTTP failures, and `error` null with a non-2xx `status_code` and
# an error object where the completion would be -- so a body is not evidence of a completion.
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
echo "$kept/$total line(s) carry a completion"

if [ "$kept" -ne "$total" ]; then
  jq -r "$usable"' select(usable | not)
    | "  dropped \(.custom_id): status=\(.response.status_code // "none") error=\(.error // "none")"' \
    "$responses"
fi

[ "$kept" -gt 0 ] || { echo "error: no line carries a completion" >&2; exit 1; }

# Only the generation batch asks for logprobs; a judge batch has none, and reporting a
# width of 0 for it would read as a batch generated at the wrong k. One width should print.
# More than one means the batch was not generated in a single pass.
widths=$(jq -r "$usable"' select(usable and .response.body.choices[0].logprobs != null)
  | .response.body.choices[0].logprobs.content[0].top_logprobs | length' "$responses" | sort -u)
if [ -n "$widths" ]; then
  echo "rank width(s) present:"
  echo "$widths" | sed "s/^/  /"
else
  echo "no logprobs in this batch (expected for judgments)"
fi
