"""
Out-of-bag bootstrap evaluation of an EPR or WEPR calibration.

Usage:
    uv run scripts/evaluate_calibration.py \
        --responses responses.jsonl --judgments judgments.jsonl --reduction wepr

Reproduces the ECIR evaluation: resample the labelled set with replacement, fit on the
sample, score the examples that fell out of it, repeat, average. Reported as the mean the
paper quotes plus a percentile interval, which is what says whether a gap between two
detectors is worth believing.

Inputs are the same two `vllm run-batch` outputs `train_calibration.py` consumes, joined
on `custom_id`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.utils import resample

from artefactual.scoring import epr, wepr
from artefactual.scoring.base_detector import DEFAULT_K
from artefactual.scoring.entropy_methods.entropy_transformer import STRATEGIES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_calibration import join_on_custom_id, read_batch_output  # noqa: E402

logger = logging.getLogger(__name__)

FACTORIES = {"epr": epr, "wepr": wepr}
SEED = 42
REPETITIONS = 1000


def out_of_bag_splits(y: np.ndarray, n_repetitions: int, seed: int):
    """Yield (train, out-of-bag) index pairs, as scikit-learn accepts for `cv`.

    Each repetition draws `n_samples` indices with replacement; whatever was never drawn
    becomes the test fold. Folds that cannot support a ranking metric -- fewer than two
    held-out examples, or only one class among them -- are skipped, because `roc_auc`
    is undefined there and scoring them would emit nan rather than a number.
    """
    indices = np.arange(len(y))
    skipped = 0
    for repetition in range(n_repetitions):
        # sklearn's own bootstrap draw; seeded per repetition so the stream is reproducible
        train = resample(indices, replace=True, n_samples=len(indices), random_state=seed + repetition)
        held_out = np.setdiff1d(indices, np.unique(train))

        if len(held_out) < 2 or len(np.unique(y[held_out])) < 2:
            skipped += 1
            continue
        yield train, held_out

    if skipped:
        logger.warning(f"skipped {skipped}/{n_repetitions} repetition(s) whose out-of-bag set could not be scored")


def evaluate(reduction: str, x, y: np.ndarray, k: int, n_repetitions: int, seed: int) -> dict:
    """Bootstrap the classifier, reusing scikit-learn for the fitting and scoring."""
    detector = FACTORIES[reduction](k=k, trainable=True)

    # The parser and entropy steps are stateless, so run them once over the whole batch
    # and bootstrap only the classifier -- otherwise every repetition re-parses the JSON.
    features = x
    for _, step in detector.steps[:-1]:
        features = step.fit_transform(features)
    features = np.asarray(features)

    if len(features) != len(y):
        msg = f"{len(features)} sequence(s) parsed but {len(y)} label(s) given; they must correspond."
        raise ValueError(msg)

    splits = list(out_of_bag_splits(y, n_repetitions, seed))
    if not splits:
        msg = "No out-of-bag fold held two classes; the dataset is too small or too imbalanced to bootstrap."
        raise ValueError(msg)

    scores = cross_validate(
        detector.steps[-1][1],
        features,
        y,
        cv=splits,
        scoring=["roc_auc", "average_precision"],
        return_estimator=True,
        n_jobs=-1,
    )

    coefficients = np.array([estimator.coef_[0] for estimator in scores["estimator"]])
    intercepts = np.array([estimator.intercept_[0] for estimator in scores["estimator"]])

    return {
        "reduction": reduction,
        "k": k,
        "n_samples": int(len(y)),
        "n_hallucinations": int(y.sum()),
        "repetitions": len(splits),
        "roc_auc": summarise(scores["test_roc_auc"]),
        "pr_auc": summarise(scores["test_average_precision"]),
        "mean_intercept": float(intercepts.mean()),
        "mean_coefficients": coefficients.mean(axis=0).tolist(),
    }


def summarise(values: np.ndarray, confidence: float = 95.0) -> dict[str, float]:
    tail = (100.0 - confidence) / 2.0
    return {
        "mean": float(np.mean(values)),
        "low": float(np.percentile(values, tail)),
        "high": float(np.percentile(values, 100.0 - tail)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=Path, required=True, help="vllm run-batch output with logprobs")
    parser.add_argument("--judgments", type=Path, required=True, help="vllm run-batch output with judge verdicts")
    parser.add_argument("--reduction", choices=sorted(STRATEGIES), default="epr", help="scoring variant")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="top_logprobs the batch was generated with")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS, help="bootstrap repetitions")
    parser.add_argument("--seed", type=int, default=SEED, help="resampling seed")
    parser.add_argument("--output", type=Path, help="where to write the report as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    x, y = join_on_custom_id(read_batch_output(args.responses), read_batch_output(args.judgments))
    report = evaluate(args.reduction, x, y, args.k, args.repetitions, args.seed)

    logger.info(f"{report['reduction']} over {report['repetitions']} bootstrap repetitions")
    for metric in ("roc_auc", "pr_auc"):
        stats = report[metric]
        logger.info(f"  {metric:>8}: {stats['mean']:.4f}  [{stats['low']:.4f}, {stats['high']:.4f}]")

    if args.output:
        args.output.write_text(json.dumps(report, indent=4), encoding="utf-8")
        logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
