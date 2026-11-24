"""Characterization of `EPR` / `WEPR` scores and load errors, pinned as literals.

Pins, for both detectors built with `from_pretrained` on a local skops file:

- `predict_proba(...)[:, 1]` and `predict_token_proba` on the checked-in Responses payload
  (docs/examples/open_ai_responses_top15.json) and on three chat-completion bodies taken
  from docs/examples/responses_sample.jsonl;
- the `ValueError` raised when the file's width does not match `k`;
- that `save_pretrained` -> `from_pretrained` reproduces the scores exactly.

The weights are fixed, hand-written logistic-regression coefficients at k=15 (1 feature
for EPR, 30 for WEPR), so no download and no randomness is involved.

The expected values are literals, not recomputed through the code under test: a check
that derives its expectation from the implementation passes whatever the implementation
does, so it could not show that a refactor left the scores unchanged. NaN marks padded
token positions and is compared as NaN.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import fitted_logistic, write_estimator

from artefactual.scoring import EPR, WEPR

EXAMPLES = Path(__file__).parent.parent / "docs" / "examples"
K = 15
NAN = float("nan")
TOLERANCE = {"rtol": 0, "atol": 1e-12}

RESPONSES = json.loads((EXAMPLES / "open_ai_responses_top15.json").read_text(encoding="utf-8"))["responses"]
# Batch lines carrying a rejection have no body, so the first three that do are taken.
CHAT_COMPLETIONS = [
    body
    for body in (
        (json.loads(line).get("response") or {}).get("body")
        for line in (EXAMPLES / "responses_sample.jsonl").read_text(encoding="utf-8").splitlines()
    )
    if body
][:3]

WEIGHTS = {
    "epr": lambda: fitted_logistic(-0.25, [0.8]),
    "wepr": lambda: fitted_logistic(0.1, [(-1) ** i * (0.05 + 0.03 * i) for i in range(2 * K)]),
}
DETECTORS = {"epr": EPR, "wepr": WEPR}

# (detector, payload) -> (predict_proba[:, 1], predict_token_proba[..., 0])
GOLDEN = {
    ("epr", "responses"): (
        [0.4380217042791768, 0.6150393962832567],
        [
            [0.4380217042791768, 0.4380217042791768, 0.4380217042791768] + [NAN] * 8,
            [
                0.437844672779974,
                0.437844672779974,
                0.6352141505664566,
                0.8456116639163577,
                0.6352141505664566,
                0.8428789370891364,
                0.437844672779974,
                0.437844672779974,
                0.6352141505664566,
                0.6352141505664566,
                0.6352141505664566,
            ],
        ],
    ),
    ("epr", "chat"): (
        [0.5174730035254576, 0.6032175967021008, 0.5155347958513531],
        [[0.5174730035254576], [0.6032175967021008], [0.5155347958513531]],
    ),
    ("wepr", "responses"): (
        [0.5248674488633032, 0.49921292859052285],
        [
            [0.5248674488633032, 0.5248674488633032, 0.5248674488633032] + [NAN] * 8,
            [
                0.524968336355548,
                0.524968336355548,
                0.5086956102879636,
                0.5037963584676807,
                0.5086956102879636,
                0.5037713284469714,
                0.524968336355548,
                0.524968336355548,
                0.5086956102879636,
                0.5086956102879636,
                0.5086956102879636,
            ],
        ],
    ),
    ("wepr", "chat"): (
        [0.5188953161936904, 0.5110103950802989, 0.5202194022561887],
        [[0.5188953161936904], [0.5110103950802989], [0.5202194022561887]],
    ),
}
PAYLOADS = {"responses": RESPONSES, "chat": CHAT_COMPLETIONS}

CASES = pytest.mark.parametrize(("name", "payload"), list(GOLDEN), ids=lambda value: str(value))


def load(name, directory):
    path = write_estimator(directory, f"{name}.skops", WEIGHTS[name]())
    return DETECTORS[name].from_pretrained(str(path), k=K)


@CASES
def test_predict_proba_is_pinned(tmp_path, name, payload):
    scores = load(name, tmp_path).predict_proba(PAYLOADS[payload])

    np.testing.assert_allclose(scores[:, 1], GOLDEN[name, payload][0], **TOLERANCE)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=0, atol=1e-12)


@CASES
def test_predict_token_proba_is_pinned(tmp_path, name, payload):
    scores = load(name, tmp_path).predict_token_proba(PAYLOADS[payload])

    expected = np.array(GOLDEN[name, payload][1])[..., None]
    assert scores.shape == expected.shape
    np.testing.assert_allclose(scores, expected, equal_nan=True, **TOLERANCE)


@pytest.mark.parametrize(
    ("name", "wrong_features", "message"),
    [
        ("epr", 2, r"takes 2 feature\(s\), but k=15 needs 1\. .*pass k=2,"),
        ("wepr", 1, r"takes 1 feature\(s\), but k=15 needs 30\. .*pass k=0,"),
        ("wepr", 20, r"takes 20 feature\(s\), but k=15 needs 30\. .*pass k=10,"),
    ],
)
def test_a_width_that_does_not_match_k_is_refused(tmp_path, name, wrong_features, message):
    path = write_estimator(tmp_path, "wrong.skops", fitted_logistic(0.0, [1.0] * wrong_features))

    with pytest.raises(ValueError, match=message) as refusal:
        DETECTORS[name].from_pretrained(str(path), k=K)
    assert DETECTORS[name].__name__ in str(refusal.value)
    assert str(path) in str(refusal.value)


@CASES
def test_saving_and_reloading_keeps_the_scores(tmp_path, name, payload):
    detector = load(name, tmp_path)
    written = detector.save_pretrained(tmp_path / "saved" / "model.skops")

    restored = DETECTORS[name].from_pretrained(str(written), k=K)

    assert np.array_equal(restored.predict_proba(PAYLOADS[payload]), detector.predict_proba(PAYLOADS[payload]))
    np.testing.assert_allclose(restored.predict_proba(PAYLOADS[payload])[:, 1], GOLDEN[name, payload][0], **TOLERANCE)
    np.testing.assert_array_equal(
        restored.predict_token_proba(PAYLOADS[payload]), detector.predict_token_proba(PAYLOADS[payload])
    )
