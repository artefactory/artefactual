#!/usr/bin/env bash
#
# Print the judge's reply from every line of a batch output file that carries one.
#
# Usage:
#   verdicts.sh <judgments.jsonl>
#
# One reply per line, in file order, as the judge wrote it -- `{"judgment": ..., ...}` when
# it followed the prompt, whatever it said instead when it did not. Lines that failed are
# skipped silently; `check_responses.sh` is what accounts for those.
#
# Piping this to `grep -c '"judgment": *true'` gives the class balance, which is the number
# step 1 is tuned against: a pack nothing hallucinates on cannot be fit.
set -euo pipefail

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit 0
fi

judgments=${1:?usage: verdicts.sh judgments.jsonl}
[ -f "$judgments" ] || { echo "error: no such file: $judgments" >&2; exit 1; }

# A body is not evidence of a completion: a rejected request carries an error object where
# the completion would be. The same rule build_judge_requests.sh and check_responses.sh
# apply, including refusing an envelope with no status_code rather than reading it as a
# success.
read -r -d '' usable <<'JQ' || true
def usable:
  if .error != null or .response == null then false
  else
    (.response.status_code
     // error("custom_id \(.custom_id): response envelope carries no status_code")) as $status
    | $status >= 200 and $status < 300 and .response.body != null
  end;
JQ

jq -r "$usable"' select(usable) | .response.body.choices[0].message.content' "$judgments"
