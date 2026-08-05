"""
Calibration training for the epr and wepr detectors.

Usage:
    python scripts/train_calibration.py \
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

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.base_detector import DEFAULT_K, BaseDetector
from artefactual.scoring.entropy_methods.entropy_transformer import STRATEGIES, EntropyTransformer

logger = logging.getLogger(__name__)


def read_batch_output(path: Path) -> dict[str, Any]:
    """Index a `vllm run-batch` output file by `custom_id`.

    Lines whose request failed carry `error` and a null `response`; they are dropped and
    counted rather than crashing the run, because one bad row should not cost a batch.
    """
    rows, failed = {}, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # `response` IS the ChatCompletion -- vLLM does not nest it under `body`.
        if record.get("error") is not None or record.get("response") is None:
            failed += 1
            continue
        rows[record["custom_id"]] = record["response"]
    if failed:
        logger.warning(f"{path.name}: dropped {failed} failed request(s)")
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
        logger.warning(
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
        logger.warning(f"dropped {unparsed} verdict(s) that could not be parsed")

    logger.info(f"joined {len(x)} pairs on custom_id ({sum(y)} hallucinations)")
    return x, np.array(y)


def train_calibration(x: Any, y: np.ndarray, reduction: str | Callable, k: int = DEFAULT_K) -> BaseDetector:
    """
    Fit an EPR or WEPR pipeline on raw completion responses.

    Args:
        x: Completion responses, in any shape `LogProbParser` accepts.
        y: Binary labels, one per generated sequence - 1 if hallucination, 0 if correct.
        reduction: scoring variant - "epr", "wepr", or a custom reduction callable.
        k: rank count to align the responses to; must be the top_logprobs used to generate.

    Returns:
        The fitted BaseDetector pipeline.
    """
    # Checked up front: EntropyTransformer only rejects a bad reduction once the whole
    # batch has been parsed, which is a long wait to be told about a typo.
    if not callable(reduction) and reduction not in STRATEGIES:
        msg = f"Invalid reduction: {reduction!r}. Expected one of {sorted(STRATEGIES)}, or a callable."
        raise ValueError(msg)

    # Assembled here rather than through epr()/wepr(): those only ever hand back a
    # calibrated detector, and an unfitted one must not escape the library.
    clf = BaseDetector(
        steps=[
            ("parser", LogProbParser()),
            ("entropy", EntropyTransformer(reduction=reduction, k=k)),
            ("classifier", LogisticRegression(penalty=None, max_iter=1000)),
        ]
    )
    clf.fit(x, y)
    return clf


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=Path, required=True, help="vllm run-batch output with logprobs")
    parser.add_argument("--judgments", type=Path, required=True, help="vllm run-batch output with judge verdicts")
    parser.add_argument("--reduction", choices=sorted(STRATEGIES), default="epr", help="scoring variant")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="top_logprobs the batch was generated with")
    parser.add_argument("--output", type=Path, help="where to write the fitted weights JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    x, y = join_on_custom_id(read_batch_output(args.responses), read_batch_output(args.judgments))
    detector = train_calibration(x, y, args.reduction, k=args.k)

    payload = weights_payload(detector, args.reduction, args.k)
    logger.info(f"intercept: {payload['intercept']}")
    logger.info(f"coefficients: {payload['coefficients']}")

    if args.output:
        args.output.write_text(json.dumps(payload, indent=4), encoding="utf-8")
        logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
