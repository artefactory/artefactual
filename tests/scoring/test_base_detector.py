import json
from importlib import resources
from pathlib import Path

import numpy as np

from artefactual.scoring.base_detector import BaseDetector, epr, wepr

# Load package weights
_DATA = resources.files("artefactual.data")
EPR_WEIGHTS = _DATA / "calibration_ministral.json"
WEPR_WEIGHTS = _DATA / "weights_mistral_small.json"

TESTS_DIR = Path(__file__).resolve().parents[1]
OPENAI_RESPONSES_PATH = TESTS_DIR / "open_ai_responses.json"

with OPENAI_RESPONSES_PATH.open("r", encoding="utf-8") as f:
    OPENAI_RESPONSES = json.load(f)

SINGLE_OPENAI_RESPONSE = OPENAI_RESPONSES["responses"][0]


def test_epr_returns_base_detector():
    assert isinstance(epr(str(EPR_WEIGHTS)), BaseDetector)


def test_wepr_returns_base_detector():
    assert isinstance(wepr(str(WEPR_WEIGHTS)), BaseDetector)


def test_epr_step_names():
    assert [name for name, _ in epr(str(EPR_WEIGHTS)).steps] == ["parser", "entropy", "classifier"]


def test_epr_entropy_reduction():
    assert epr(str(EPR_WEIGHTS)).named_steps["entropy"].reduction == "epr"


def test_wepr_entropy_reduction():
    assert wepr(str(WEPR_WEIGHTS)).named_steps["entropy"].reduction == "wepr"


def test_epr_with_pretrained_has_coef():
    clf = epr(str(EPR_WEIGHTS)).named_steps["classifier"]
    assert clf.coef_.shape == (1, 1)  # 1 class, 1 feature (mean_entropy)


def test_from_pretrained_epr():
    detector = BaseDetector.from_pretrained(str(EPR_WEIGHTS), reduction="epr")
    assert isinstance(detector, BaseDetector)
    assert detector.named_steps["classifier"].coef_ is not None


def test_predict_proba_output_shape():
    scores = epr(str(EPR_WEIGHTS)).predict_proba(SINGLE_OPENAI_RESPONSE)
    assert scores.shape == (1, 2)  # 1 sequence, 2 classes


def test_predict_proba_valid_probabilities():
    scores = epr(str(EPR_WEIGHTS)).predict_proba(SINGLE_OPENAI_RESPONSE)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_unfitted_detector_returns_raw_entropy():
    scores = epr(str(EPR_WEIGHTS)).predict_proba(SINGLE_OPENAI_RESPONSE)
    assert scores.shape == (1, 2)
    assert scores[0, 0] == 0.0  # column 0 is always zero in uncalibrated mode
    assert scores[0, 1] > 0.0  # column 1 is raw mean entropy


def test_predict_token_proba_shape():
    token_scores = epr(str(EPR_WEIGHTS)).predict_token_proba(SINGLE_OPENAI_RESPONSE)
    assert token_scores.shape[0] == 1  # 1 sequence
    assert token_scores.shape[2] == 1


def test_predict_token_proba_valid_scores():
    token_scores = epr(str(EPR_WEIGHTS)).predict_token_proba(SINGLE_OPENAI_RESPONSE)
    valid = token_scores[~np.isnan(token_scores)]
    assert len(valid) > 0
    assert np.all(valid >= 0) and np.all(valid <= 1)
