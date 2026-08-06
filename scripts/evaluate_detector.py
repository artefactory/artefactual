"""
Cross-validated evaluation of an EPR or WEPR detector.

Usage:
    uv run scripts/evaluate_detector.py \
        --responses responses.jsonl --judgments judgments.jsonl --reduction wepr

Reports ROC-AUC and PR-AUC across stratified folds, plus a per-class classification report.
Inputs are the same two `vllm run-batch` outputs `train_detector.py` consumes, joined on
`custom_id`.

Stratification is the point: hallucinations are the rare class, and an unstratified split
of an imbalanced set produces folds holding only one class, where ROC-AUC is undefined and
`LogisticRegression.fit` refuses outright. `StratifiedKFold` keeps the class ratio in every
fold, so the run either works on all of them or fails once, up front, naming the data.

This is not the paper's estimator. The paper bootstraps -- resample with replacement, score
what fell out, repeat -- and quotes a percentile interval. A k-fold mean and standard
deviation is close in spirit but not the same statistic, so the spread here is not directly
comparable to the intervals in Table 1.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from absl import app, flags, logging
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from artefactual.scoring import epr, wepr
from artefactual.scoring.base_detector import DEFAULT_K
from artefactual.scoring.entropy_methods.entropy_transformer import STRATEGIES

FACTORIES = {"epr": epr, "wepr": wepr}
SEED = 42
FOLDS = 5
LABEL_NAMES = ["grounded", "hallucination"]

FLAGS = flags.FLAGS

flags.DEFINE_string("responses", None, "vllm run-batch output with logprobs")
flags.DEFINE_string("judgments", None, "vllm run-batch output with judge verdicts")
flags.DEFINE_enum("reduction", "epr", sorted(STRATEGIES), "scoring variant")
flags.DEFINE_integer("k", DEFAULT_K, "top_logprobs the batch was generated with")
flags.DEFINE_integer("folds", FOLDS, "stratified cross-validation folds")
flags.DEFINE_integer("seed", SEED, "fold shuffling seed")
flags.DEFINE_string("output", None, "where to write the report as JSON")

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


def evaluate(reduction: str, x, y: np.ndarray, k: int, n_folds: int, seed: int) -> dict:
    """Cross-validate the classifier, reporting ranking metrics and a per-class breakdown."""
    detector = FACTORIES[reduction](k=k, trainable=True)

    # The parser and entropy steps are stateless, so run them once over the whole batch and
    # cross-validate only the classifier -- otherwise every fold re-parses the JSON.
    features = x
    for _, step in detector.steps[:-1]:
        features = step.fit_transform(features)
    features = np.asarray(features)

    if len(features) != len(y):
        msg = f"{len(features)} sequence(s) parsed but {len(y)} label(s) given; they must correspond."
        raise ValueError(msg)

    # Checked here rather than left to sklearn: StratifiedKFold only warns when a class has
    # fewer members than folds, then hands a single-class training set to a solver that
    # raises on it -- a traceback from inside a fold, naming neither the class nor the count.
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2 or counts.min() < n_folds:
        msg = (
            f"{n_folds}-fold cross-validation needs at least {n_folds} of each class, but the "
            f"labels are {dict(zip(labels.tolist(), counts.tolist(), strict=True))} "
            f"(0 = grounded, 1 = hallucination). Label more data, or lower --folds."
        )
        raise ValueError(msg)

    classifier = detector.steps[-1][1]
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    scores = cross_validate(classifier, features, y, cv=folds, scoring=["roc_auc", "average_precision"])
    predictions = cross_val_predict(classifier, features, y, cv=folds)

    report_kwargs = {"target_names": LABEL_NAMES, "zero_division": 0}
    logging.info("\n%s", classification_report(y, predictions, **report_kwargs))

    return {
        "reduction": reduction,
        "k": k,
        "n_samples": len(y),
        "n_hallucinations": int(y.sum()),
        "folds": n_folds,
        "roc_auc": summarise(scores["test_roc_auc"]),
        "pr_auc": summarise(scores["test_average_precision"]),
        "classification_report": classification_report(y, predictions, output_dict=True, **report_kwargs),
    }


def summarise(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        msg = f"unexpected positional argument(s): {argv[1:]}"
        raise app.UsageError(msg)

    x, y = join_on_custom_id(read_batch_output(Path(FLAGS.responses)), read_batch_output(Path(FLAGS.judgments)))
    report = evaluate(FLAGS.reduction, x, y, FLAGS.k, FLAGS.folds, FLAGS.seed)

    logging.info(f"{report['reduction']} over {report['folds']} stratified folds")
    for metric in ("roc_auc", "pr_auc"):
        stats = report[metric]
        logging.info(f"  {metric:>8}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

    if FLAGS.output:
        Path(FLAGS.output).write_text(json.dumps(report, indent=4), encoding="utf-8")
        logging.info(f"wrote {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
