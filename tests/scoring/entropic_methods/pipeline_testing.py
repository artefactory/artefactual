from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from artefactual.preprocessing.parser import LogProbParser
from artefactual.scoring.entropy_methods.entropy_transformer import EPRFeatureExtractor, WEPRFeatureExtractor
from artefactual.scoring.pretrained_regression import PretrainedLogisticRegression
from examples.mock_vllm import load_json

mock_reponses = load_json(Path(__file__).parent.parent / "examples/wepr_demo_responses.json")


def epr_pipeline():
    return Pipeline([
        ("parser", LogProbParser()),
        ("features", EPRFeatureExtractor(k=15)),
        (
            "detector",
            PretrainedLogisticRegression.from_pretrained("src/artefactual/data/calibration_mistral_small.json"),
        ),
    ])


def wepr_pipeline():
    return Pipeline([
        ("parser", LogProbParser()),
        ("features", WEPRFeatureExtractor(k=15)),
        ("detector", PretrainedLogisticRegression.from_pretrained("src/artefactual/data/weights_mistral_small.json")),
    ])


def test_epr(epr_pipeline):
    proba = epr_pipeline.predict_proba(mock_reponses)
    assert proba.shape == (1, 2)  # test epr pipeline output shape
    assert np.allclose(proba.sum(axis=1), 1.0)  # test epr pipeline probabilities sum to one
    assert (proba >= 0).all() and (proba <= 1).all()  # test epr pipeline probabilities in range
    return proba


epr_pipeline_instance = epr_pipeline()
# print(test_epr(epr_pipeline_instance))


def test_wepr(wepr_pipeline):
    proba = wepr_pipeline.predict_proba(mock_reponses)
    assert proba.shape == (1, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()
    return proba


wepr_pipeline_instance = wepr_pipeline()
# print(test_wepr(wepr_pipeline_instance))
