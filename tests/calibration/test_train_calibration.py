import json
from pathlib import Path

import numpy as np
import pytest

from artefactual.calibration import train_calibration
from artefactual.scoring.base_detector import BaseDetector

FIXTURE_PATH = Path(__file__).parent.parent / "open_ai_responses.json"


@pytest.fixture
def x_train():
    with Path(FIXTURE_PATH).open() as f:
        responses = json.load(f)["responses"]
    return {"object": "response", "output": responses[0]["output"] + responses[1]["output"]}


def test_train_calibration_returns_fitted_pipeline(x_train):
    clf = train_calibration(x_train, np.array([0, 1]), "wepr")
    assert isinstance(clf, BaseDetector)


def test_train_calibration_epr(x_train):
    clf = train_calibration(x_train, np.array([0, 1]), "epr")
    assert isinstance(clf, BaseDetector)


def test_train_calibration_invalid_reduction():
    with pytest.raises(ValueError, match="Invalid reduction"):
        train_calibration([], np.array([]), "invalid")


def test_train_calibration_mismatched_lengths():
    with pytest.raises(ValueError, match="Length of x"):
        train_calibration([1, 2, 3], np.array([0, 1]), "epr")


def test_train_calibration_insufficient_classes():
    with pytest.raises(ValueError, match="Need both positive and negative labels"):
        train_calibration([1, 2], np.array([0, 0]), "epr")
