"""
Trains a hallucination detector and evaluates the model it produces.

Usage:
    uv run scripts/train_detector.py \
        --responses responses.jsonl --judgments judgments.jsonl --reduction wepr \
        --output wepr.skops --report wepr_evaluation.json

Both inputs are `vllm run-batch` outputs: one line per request, carrying the `custom_id`
the generation request was built with. Rows are joined on that id rather than on position,
so a reordered or partially failed batch cannot silently train on mismatched pairs.

`--responses` holds the generations, whose `logprobs`/`top_logprobs` the detector scores.
`--judgments` holds the LLM-as-a-judge verdicts, as JSON `{"judgment": true|false}` in the
message content; `judgment: true` means the answer was correct, so the training label is
its negation -- 1 marks a hallucination. `read_judgment` owns that reading, including the
fenced and prose-wrapped replies a model actually returns.

The labelled set is split once, stratified: the detector is fitted on the training part and
then scored on the held-out part. The numbers therefore describe *the model in `--output`*,
not an average over models that were refitted and thrown away -- so what is reported is what
is shipped. `--test-size 0` skips the split and the evaluation, fitting on everything.

Confidence intervals come from resampling the held-out predictions, following the paper in
reporting a mean with a 95% percentile interval, because a mean alone cannot say whether a
gap between two detectors is real. The paper resamples the whole fit; this resamples one
model's scores, which answers "how precisely do I know this model's ROC-AUC" rather than
"how much does the fitting procedure vary". The intervals are therefore narrower than
Table 1's and not directly comparable to them.

The fitted estimator is written to `--output` as a `.skops` file, the shape
`epr()` and `wepr()` read back; the evaluation report goes to `--report` as JSON.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from absl import app, flags, logging
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from artefactual.preprocessing import index_by_custom_id, read_batch, read_judgment
from artefactual.scoring import BaseDetector
from artefactual.scoring.base_detector import DEFAULT_K
from artefactual.scoring.entropy_methods.entropy_transformer import STRATEGIES

SEED = 42
REPETITIONS = 1000
TEST_SIZE = 0.25
LABEL_NAMES = ["grounded", "hallucination"]

FLAGS = flags.FLAGS

flags.DEFINE_string("responses", None, "vllm run-batch output with logprobs")
flags.DEFINE_string("judgments", None, "vllm run-batch output with judge verdicts")
flags.DEFINE_enum("reduction", "epr", sorted(STRATEGIES), "scoring variant")
flags.DEFINE_integer("k", DEFAULT_K, "top_logprobs the batch was generated with")
flags.DEFINE_string("output", None, "where to write the fitted detector (.skops)")
flags.DEFINE_string("report", None, "where to write the evaluation as JSON")
flags.DEFINE_float("test_size", TEST_SIZE, "held-out fraction; 0 fits on everything and skips evaluation")
flags.DEFINE_integer("repetitions", REPETITIONS, "resamples used for the confidence intervals")
flags.DEFINE_integer("seed", SEED, "seed for the split and the resampling")

flags.mark_flags_as_required(["responses", "judgments"])


def read_responses(path: Path) -> dict[str, Any]:
    """Index a `vllm run-batch` output file by `custom_id`, dropping the failed requests.

    `read_batch` owns the shape and the duplicate-`custom_id` refusal. What is added here is
    the count: a line whose request failed carries no completion, and dropping it silently
    would hide a batch that half succeeded.
    """
    rows = read_batch(path)
    kept = index_by_custom_id(rows)
    if failed := len(rows) - len(kept):
        logging.warning(f"{path.name}: dropped {failed} failed request(s)")
    return {custom_id: row.completion for custom_id, row in kept.items()}


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
        judgment = read_judgment(judgments[custom_id])
        if judgment is None:
            unparsed += 1
            continue
        x.append(responses[custom_id])
        y.append(0 if judgment else 1)  # judgment True == correct answer == not a hallucination
    if unparsed:
        logging.warning(f"dropped {unparsed} verdict(s) that could not be parsed")

    logging.info(f"joined {len(x)} pairs on custom_id ({sum(y)} hallucinations)")
    return x, np.array(y)


def bootstrap_interval(y_true: np.ndarray, scores: np.ndarray, metric, n_repetitions: int, seed: int) -> dict:
    """Mean and 95% percentile interval for a ranking metric, by resampling the held-out set.

    The model is fixed; what is resampled is which held-out examples were drawn. That is the
    question a shipped detector raises -- how precisely its score is known -- and it is why
    the interval can be reported for the exact artefact written to `--output`.

    Resamples that end up single-class are skipped: `roc_auc` is undefined on them. With a
    rare positive class that can be most of them, so the count of survivors is returned and
    should be read alongside the interval.
    """
    indices = np.arange(len(y_true))
    values = []
    for repetition in range(n_repetitions):
        drawn = resample(indices, replace=True, n_samples=len(indices), random_state=seed + repetition)
        if len(np.unique(y_true[drawn])) < 2:
            continue
        values.append(metric(y_true[drawn], scores[drawn]))

    if not values:
        return {"point": float(metric(y_true, scores)), "mean": None, "low": None, "high": None, "resamples": 0}

    values = np.asarray(values)
    return {
        "point": float(metric(y_true, scores)),
        "mean": float(values.mean()),
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
        "resamples": len(values),
    }


def evaluate(detector: BaseDetector, x_test, y_test: np.ndarray, n_repetitions: int, seed: int) -> dict:
    """Score the fitted detector on the held-out set it has never seen."""
    scores = detector.predict_proba(x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)

    report_kwargs = {"target_names": LABEL_NAMES, "zero_division": 0}
    logging.info("\n%s", classification_report(y_test, predictions, **report_kwargs))

    return {
        "n_test": len(y_test),
        "n_hallucinations": int(y_test.sum()),
        "roc_auc": bootstrap_interval(y_test, scores, roc_auc_score, n_repetitions, seed),
        "pr_auc": bootstrap_interval(y_test, scores, average_precision_score, n_repetitions, seed),
        "classification_report": classification_report(y_test, predictions, output_dict=True, **report_kwargs),
    }


def split_labelled_set(x, y: np.ndarray, test_size: float, seed: int) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Stratified holdout, refusing splits that cannot carry both classes.

    Checked here rather than left to sklearn, which raises about `n_splits` and group counts
    without saying which class is short or by how much.
    """
    labels, counts = np.unique(y, return_counts=True)
    balance = dict(zip(labels.tolist(), counts.tolist(), strict=True))
    if len(labels) < 2:
        msg = f"Both classes are needed to fit a detector, but the labels are {balance} (0 = grounded, 1 = hallucination)."
        raise ValueError(msg)
    if min(counts) < 2:
        msg = (
            f"A stratified holdout needs at least 2 of each class, but the labels are {balance} "
            f"(0 = grounded, 1 = hallucination). Label more data, or pass --test_size 0 to fit "
            f"without evaluating."
        )
        raise ValueError(msg)

    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=seed)


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        msg = f"unexpected positional argument(s): {argv[1:]}"
        raise app.UsageError(msg)

    x, y = join_on_custom_id(read_responses(Path(FLAGS.responses)), read_responses(Path(FLAGS.judgments)))

    if FLAGS.test_size > 0:
        x_train, x_test, y_train, y_test = split_labelled_set(x, y, FLAGS.test_size, FLAGS.seed)
        logging.info(f"fitting on {len(y_train)} response(s), holding out {len(y_test)}")
    else:
        x_train, y_train, x_test, y_test = x, y, None, None
        logging.info(f"fitting on all {len(y)} response(s); --test_size 0, so no evaluation")

    detector = BaseDetector.trainable(FLAGS.reduction, k=FLAGS.k).fit(x_train, y_train)

    classifier = detector.named_steps["classifier"]
    logging.info(f"intercept: {float(classifier.intercept_[0])}")
    logging.info(f"coefficients: {classifier.coef_[0].tolist()}")
    if FLAGS.output:
        logging.info(f"wrote {detector.save_estimator(FLAGS.output)}")

    if y_test is None:
        return

    report = evaluate(detector, x_test, y_test, FLAGS.repetitions, FLAGS.seed)
    for metric in ("roc_auc", "pr_auc"):
        stats = report[metric]
        interval = "" if stats["mean"] is None else f"  [{stats['low']:.4f}, {stats['high']:.4f}]"
        logging.info(f"  {metric:>8}: {stats['point']:.4f}{interval}")

    if FLAGS.report:
        Path(FLAGS.report).write_text(json.dumps(report, indent=4), encoding="utf-8")
        logging.info(f"wrote {FLAGS.report}")


if __name__ == "__main__":
    app.run(main)
