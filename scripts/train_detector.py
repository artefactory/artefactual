"""
Trains an epr or wepr hallucination detector from a labelled batch.

Usage:
    uv run scripts/train_detector.py \
        --responses responses.jsonl --judgments judgments.jsonl --reduction epr

Both inputs are `vllm run-batch` outputs: one line per request, carrying the `custom_id`
the generation request was built with. Rows are joined on that id rather than on position,
so a reordered or partially failed batch cannot silently train on mismatched pairs.

`--responses` holds the generations, whose `logprobs`/`top_logprobs` the detector scores.
`--judgments` holds the LLM-as-a-judge verdicts, as JSON `{"judgment": true|false}` in the
message content; `judgment: true` means the answer was correct, so the training label is
its negation -- 1 marks a hallucination.

The fitted intercept and coefficients are written to `--output` in the shape
`artefactual.utils.io.load_weights` reads back.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from absl import app, flags, logging

from artefactual.scoring import BaseDetector, epr, wepr
from artefactual.scoring.base_detector import DEFAULT_K
from artefactual.scoring.entropy_methods.entropy_transformer import STRATEGIES

FLAGS = flags.FLAGS

flags.DEFINE_string("responses", None, "vllm run-batch output with logprobs")
flags.DEFINE_string("judgments", None, "vllm run-batch output with judge verdicts")
flags.DEFINE_enum("reduction", "epr", sorted(STRATEGIES), "scoring variant")
flags.DEFINE_integer("k", DEFAULT_K, "top_logprobs the batch was generated with")
flags.DEFINE_string("output", None, "where to write the fitted weights JSON")

flags.mark_flags_as_required(["responses", "judgments"])


def read_batch_output(path: Path) -> dict[str, Any]:
    """Index a `vllm run-batch` output file by `custom_id`.

    Lines whose request failed carry `error` and a null `response`; they are dropped and
    counted rather than crashing the run, because one bad row should not cost a batch.

    `run-batch` follows the OpenAI Batch output spec, so `response` is an envelope --
    `{status_code, request_id, body}` -- and the ChatCompletion is its `body`. Older vllm
    put the completion directly in `response`, so unwrap only when the envelope is there.
    """
    rows, failed = {}, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("error") is not None or record.get("response") is None:
            failed += 1
            continue
        response = record["response"]
        rows[record["custom_id"]] = response.get("body", response)
    if failed:
        logging.warning(f"{path.name}: dropped {failed} failed request(s)")
    return rows


def parse_judgment(completion: Any) -> bool | None:
    """Read the judge verdict out of a completion.

    The judge is asked for `{"judgment": true/false, "explanation": ...}`. Models wrap
    that in prose or fences often enough that a bare `json.loads` is not safe, so fall
    back to scanning for the literal token.
    """
    content = completion["choices"][0]["message"]["content"]
    try:
        return bool(json.loads(content)["judgment"])
    except (json.JSONDecodeError, KeyError, TypeError):
        lowered = content.lower()
        if '"judgment": true' in lowered or lowered.strip() in {"true", "true."}:
            return True
        if '"judgment": false' in lowered or lowered.strip() in {"false", "false."}:
            return False
        return None


def join_on_custom_id(responses: dict[str, Any], judgments: dict[str, Any]) -> tuple[list[Any], np.ndarray]:
    """Pair each generation with its verdict, in a single deterministic order.

    Returns the responses and the 0/1 labels, where 1 marks a hallucination.
    """
    shared = sorted(set(responses) & set(judgments))
    if not shared:
        msg = "No custom_id is present in both files; check they came from the same batch."
        raise ValueError(msg)

    only_responses = sorted(set(responses) - set(judgments))
    only_judgments = sorted(set(judgments) - set(responses))
    if only_responses or only_judgments:
        logging.warning(
            f"{len(only_responses)} generation(s) without a verdict and "
            f"{len(only_judgments)} verdict(s) without a generation were dropped"
        )

    x, y, unparsed = [], [], 0
    for custom_id in shared:
        judgment = parse_judgment(judgments[custom_id])
        if judgment is None:
            unparsed += 1
            continue
        x.append(responses[custom_id])
        y.append(0 if judgment else 1)  # judgment True == correct answer == not a hallucination
    if unparsed:
        logging.warning(f"dropped {unparsed} verdict(s) that could not be parsed")

    logging.info(f"joined {len(x)} pairs on custom_id ({sum(y)} hallucinations)")
    return x, np.array(y)


def weights_payload(detector: BaseDetector, reduction: str, k: int) -> dict[str, Any]:
    """Serialise the fitted classifier in the shape `load_weights` reads back."""
    classifier = detector.named_steps["classifier"]
    coefficients = classifier.coef_[0].tolist()

    if reduction == "epr":
        named = {"mean_entropy": coefficients[0]}
    else:
        named = {f"mean_rank_{i + 1}": coefficients[i] for i in range(k)}
        named |= {f"max_rank_{i + 1}": coefficients[k + i] for i in range(k)}

    return {"intercept": float(classifier.intercept_[0]), "coefficients": named}


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        msg = f"unexpected positional argument(s): {argv[1:]}"
        raise app.UsageError(msg)

    x, y = join_on_custom_id(read_batch_output(Path(FLAGS.responses)), read_batch_output(Path(FLAGS.judgments)))
    detector = {"epr": epr, "wepr": wepr}[FLAGS.reduction](k=FLAGS.k, trainable=True).fit(x, y)

    payload = weights_payload(detector, FLAGS.reduction, FLAGS.k)
    logging.info(f"intercept: {payload['intercept']}")
    logging.info(f"coefficients: {payload['coefficients']}")

    if FLAGS.output:
        Path(FLAGS.output).write_text(json.dumps(payload, indent=4), encoding="utf-8")
        logging.info(f"wrote {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
