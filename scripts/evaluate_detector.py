"""
Out-of-bag bootstrap evaluation of an EPR or WEPR detector.

Usage:
    uv run scripts/evaluate_detector.py \
        --responses responses.jsonl --judgments judgments.jsonl --reduction wepr

Reproduces the paper's evaluation: resample the labelled set with replacement, fit on the
sample, score the examples that fell out of it, repeat. Reported as the mean the paper
quotes plus a 95% percentile interval, which is what says whether a gap between two
detectors is worth believing. A per-class classification report is printed alongside.

Inputs are the same two `vllm run-batch` outputs `train_detector.py` consumes, joined on
`custom_id`.

The resampling lives in `OutOfBagBootstrap`, a `BaseCrossValidator` defined below, because
scikit-learn ships no bootstrap splitter: every splitter it provides samples without
replacement, and `cross_validation.Bootstrap` was removed long ago. Writing it as a splitter
keeps the rest of the script ordinary scikit-learn -- `cross_validate` does the fitting and
scoring. The one alternative, `BaggingClassifier(oob_score=True)`, yields a single aggregate
number rather than a distribution, so it cannot produce the interval the paper reports.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from absl import app, flags, logging
from sklearn.metrics import classification_report
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.utils import resample

from artefactual.scoring import epr, wepr
from artefactual.scoring.base_detector import DEFAULT_K
from artefactual.scoring.entropy_methods.entropy_transformer import STRATEGIES

FACTORIES = {"epr": epr, "wepr": wepr}
SEED = 42
REPETITIONS = 1000
LABEL_NAMES = ["grounded", "hallucination"]

FLAGS = flags.FLAGS

flags.DEFINE_string("responses", None, "vllm run-batch output with logprobs")
flags.DEFINE_string("judgments", None, "vllm run-batch output with judge verdicts")
flags.DEFINE_enum("reduction", "epr", sorted(STRATEGIES), "scoring variant")
flags.DEFINE_integer("k", DEFAULT_K, "top_logprobs the batch was generated with")
flags.DEFINE_integer("repetitions", REPETITIONS, "bootstrap repetitions")
flags.DEFINE_integer("seed", SEED, "resampling seed")
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


class OutOfBagBootstrap(BaseCrossValidator):
    """Bootstrap resampling as a scikit-learn splitter: train on the draw, test on the rest.

    Each repetition draws `n_samples` indices with replacement; whatever was never drawn is
    the test fold, which is on average 1/e of the set. This is the scheme the paper
    evaluates with, and scikit-learn has no equivalent -- every splitter it ships samples
    without replacement, and `cross_validation.Bootstrap` was removed long ago. Written as a
    splitter rather than a helper so it drops into `cross_validate`, `cross_val_score` or a
    search unchanged.

    Draws that cannot be fitted or cannot be scored are skipped, which matters because
    hallucination labels are imbalanced: a resample of a rare class can miss it entirely and
    `LogisticRegression.fit` raises on single-class input, while `roc_auc` is undefined on an
    out-of-bag set holding fewer than two examples or only one class. Skipping reports an
    interval over the draws that worked instead of aborting on the first unlucky one.

    `y` is therefore required -- the class-balance checks cannot be made without it.

    Args:
        n_repetitions: Draws to attempt. Skipped ones are not retried, so the number of
            splits actually yielded can be lower.
        random_state: Offset for the per-repetition seed, so a rerun reproduces the folds.
    """

    def __init__(self, n_repetitions: int = 1000, random_state: int = 0) -> None:
        self.n_repetitions = n_repetitions
        self.random_state = random_state

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:  # noqa: ARG002, N803
        if y is None:
            msg = f"{type(self).__name__} needs y to check that each draw holds both classes."
            raise ValueError(msg)

        y = np.asarray(y)
        indices = np.arange(len(y))
        skipped = 0
        for repetition in range(self.n_repetitions):
            # sklearn's own bootstrap draw; seeded per repetition so the stream is reproducible
            seed = self.random_state + repetition
            train = resample(indices, replace=True, n_samples=len(indices), random_state=seed)
            held_out = np.setdiff1d(indices, np.unique(train))

            if len(np.unique(y[train])) < 2 or len(held_out) < 2 or len(np.unique(y[held_out])) < 2:
                skipped += 1
                continue
            yield train, held_out

        if skipped:
            logging.warning(f"skipped {skipped}/{self.n_repetitions} draw(s) that could not be fitted or scored")

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: ARG002, N803
        """Splits actually yielded, which is below `n_repetitions` whenever draws are skipped."""
        if y is None:
            return self.n_repetitions
        return sum(1 for _ in self.split(X, y))

    def _iter_test_indices(self, X=None, y=None, groups=None) -> Iterator[np.ndarray]:  # noqa: N803
        """Required by `BaseCrossValidator`; `split` is overridden, so this is never reached."""
        raise NotImplementedError


def out_of_bag_predictions(estimators, splits, x, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Average each example's predicted probability over the repetitions that held it out.

    A bootstrap example is out of bag many times over, so unlike k-fold there is no single
    held-out prediction per example -- `cross_val_predict` cannot be used here at all, since
    it requires each example to be predicted exactly once. Averaging the out-of-bag
    probabilities is the standard reading, and thresholding that average at 0.5 gives the
    decisions the classification report scores.

    Returns the labels mask and the 0/1 predictions for the examples that fell out at least
    once. An example drawn into every single training set has no honest prediction and is
    left out rather than scored against a model that saw it.
    """
    totals, counts = np.zeros(n_samples), np.zeros(n_samples, dtype=int)
    for estimator, (_, held_out) in zip(estimators, splits, strict=True):
        totals[held_out] += estimator.predict_proba([x[i] for i in held_out])[:, 1]
        counts[held_out] += 1

    scored = counts > 0
    if not scored.all():
        logging.warning(f"{(~scored).sum()} example(s) were never out of bag and are absent from the report")
    return scored, (totals[scored] / counts[scored] >= 0.5).astype(int)


def evaluate(reduction: str, x, y: np.ndarray, k: int, n_repetitions: int, seed: int) -> dict:
    """Bootstrap the detector, letting scikit-learn do the fitting and scoring.

    The whole `BaseDetector` pipeline is cross-validated, not just its classifier: each draw
    refits the parser and the entropy transform on that draw alone, and the out-of-bag
    responses go in raw. Pre-transforming once outside the loop would be faster, but it fits
    a step on examples the fold is meant to be held out from -- harmless while those steps
    stay stateless, and a silent leak the moment one of them learns anything.
    """
    detector = FACTORIES[reduction](k=k, trainable=True)

    if len(x) != len(y):
        msg = f"{len(x)} response(s) given but {len(y)} label(s); they must correspond."
        raise ValueError(msg)

    splits = list(OutOfBagBootstrap(n_repetitions=n_repetitions, random_state=seed).split(x, y))
    if not splits:
        labels, counts = np.unique(y, return_counts=True)
        msg = (
            f"No bootstrap repetition produced both a trainable draw and a scoreable out-of-bag "
            f"set. Labels: {dict(zip(labels.tolist(), counts.tolist(), strict=True))} "
            f"(0 = grounded, 1 = hallucination). The rarer class is too small for a resample to "
            f"reliably contain it and still leave some out -- label more data."
        )
        raise ValueError(msg)

    scores = cross_validate(
        detector,
        x,
        y,
        cv=splits,
        scoring=["roc_auc", "average_precision"],
        return_estimator=True,
        n_jobs=-1,
    )

    scored, predictions = out_of_bag_predictions(scores["estimator"], splits, x, len(y))
    report_kwargs = {"target_names": LABEL_NAMES, "zero_division": 0}
    logging.info("\n%s", classification_report(y[scored], predictions, **report_kwargs))

    return {
        "reduction": reduction,
        "k": k,
        "n_samples": len(y),
        "n_hallucinations": int(y.sum()),
        "repetitions": len(splits),
        "roc_auc": summarise(scores["test_roc_auc"]),
        "pr_auc": summarise(scores["test_average_precision"]),
        "classification_report": classification_report(y[scored], predictions, output_dict=True, **report_kwargs),
    }


def summarise(values: np.ndarray, confidence: float = 95.0) -> dict[str, float]:
    """Mean plus a percentile interval, as the paper reports it."""
    tail = (100.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "low": float(np.percentile(values, tail)),
        "high": float(np.percentile(values, 100.0 - tail)),
    }


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        msg = f"unexpected positional argument(s): {argv[1:]}"
        raise app.UsageError(msg)

    x, y = join_on_custom_id(read_batch_output(Path(FLAGS.responses)), read_batch_output(Path(FLAGS.judgments)))
    report = evaluate(FLAGS.reduction, x, y, FLAGS.k, FLAGS.repetitions, FLAGS.seed)

    logging.info(f"{report['reduction']} over {report['repetitions']} bootstrap repetitions")
    for metric in ("roc_auc", "pr_auc"):
        stats = report[metric]
        logging.info(f"  {metric:>8}: {stats['mean']:.4f}  [{stats['low']:.4f}, {stats['high']:.4f}]")

    if FLAGS.output:
        Path(FLAGS.output).write_text(json.dumps(report, indent=4), encoding="utf-8")
        logging.info(f"wrote {FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
