"""
Calibration training for epr and wepr detectors.

Usage:
    python scripts/train_calibration.py --responses responses.json --labels labels.json --reduction epr

`--responses` is a completion-response JSON, either a bare payload, a list of them,
or the `{"responses": [...]}` wrapper the generation step writes. `--labels` holds one
0/1 per generated sequence, as a JSON list or a single-column CSV.

The fitted intercept and coefficients are reported on stdout.
"""

import argparse
import csv
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from beartype.door import is_bearable
from sklearn.linear_model import LogisticRegression

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.base_detector import BaseDetector
from artefactual.scoring.entropy_methods.entropy_transformer import STRATEGIES, EntropyTransformer

logger = logging.getLogger(__name__)


def train_calibration(x: Any, y: np.ndarray, reduction: str | Callable) -> BaseDetector:
    """
    Fits an EPR or WEPR pipeline on raw JSON responses.

    Args:
        x: Raw completion responses, in any shape `LogProbParser` accepts: a single
            response payload carrying several generations, or a sequence of them.
            One training sample per generated sequence, not per element of `x`.
        y: Binary labels, one per generated sequence - 1 if hallucination, 0 if correct.
        reduction: scoring variant - "epr", "wepr", or a custom reduction callable.

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
            ("entropy", EntropyTransformer(reduction=reduction)),
            ("classifier", LogisticRegression()),
        ]
    )
    clf.fit(x, y)
    scores = clf.predict_proba(x)[:, 1]
    logger.info(f"Trained on {len(scores)} sequences. Hallucination probabilities: {scores}")
    return clf


def load_responses(path: Path) -> Any:
    """Read a response file, unwrapping the `{"responses": [...]}` shape if present."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if is_bearable(payload, dict) and "responses" in payload:
        return payload["responses"]
    return payload


def load_labels(path: Path) -> np.ndarray:
    """Read one 0/1 label per generated sequence, from a JSON list or a one-column CSV."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [row[0] for row in csv.reader(handle) if row]
        # tolerate a header, which a CSV written by pandas carries
        if rows and not rows[0].strip().lstrip("-").isdigit():
            rows = rows[1:]
        return np.array([int(value) for value in rows])
    return np.asarray(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=Path, required=True, help="completion responses carrying logprobs")
    parser.add_argument("--labels", type=Path, required=True, help="one 0/1 per sequence: JSON list or CSV column")
    parser.add_argument("--reduction", choices=sorted(STRATEGIES), default="epr", help="scoring variant")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    detector = train_calibration(load_responses(args.responses), load_labels(args.labels), args.reduction)

    classifier = detector.named_steps["classifier"]
    logger.info(f"intercept: {classifier.intercept_.tolist()}")
    logger.info(f"coefficients: {classifier.coef_.tolist()}")


if __name__ == "__main__":
    main()
