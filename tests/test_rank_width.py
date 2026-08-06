"""The rank axis of a response must match the rank count its detector was fit at.

A weights file is trained at a fixed `k`, while the rank width of a *response* is
caller-controlled: OpenAI's `top_logprobs` caps at 20 and 5 is a common default, whereas
every shipped weights file is trained at k=15. The two have to be reconciled, and the
direction matters:

* **Wider than k** is reconcilable. The surplus ranks are dropped, because the detector was
  trained on exactly k of them and has no coefficient for a k+1th.
* **Narrower than k** is not. Those ranks are not absent from the distribution, only
  unfetched. EPR sums the rank axis, so zero-filling them simply drops their contributions
  and the response scores as more confident than it was -- indistinguishable from a model
  that really was that sure. That reorders results rather than rescaling them, so the
  pipeline refuses it.

`LogProbParser` owns this: it is the only step that sees a token's true rank count, since
everything downstream receives a batch already padded to a common width. That is also why
the check is per response and not per batch -- a batch's widest member would otherwise
hide its narrowest, which is the member being scored wrongly.

Widths and coefficients are drawn rather than hand-picked, so the contract is asserted
across the whole range a caller can produce rather than at a few chosen points.
"""

import numpy as np
import pytest
from conftest import chat_payloads_of_fixed_width, estimators, payload_width, write_estimator
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from artefactual.scoring import epr, wepr

CALIBRATED_K = 15

# Below and at/above the trained width — the two sides of the contract.
narrow_payloads = chat_payloads_of_fixed_width(min_ranks=1, max_ranks=CALIBRATED_K - 1)
wide_payloads = chat_payloads_of_fixed_width(min_ranks=CALIBRATED_K, max_ranks=25)

# A detector fit at exactly the width the estimators default to.
dense_weights = estimators(n_features=2 * CALIBRATED_K)

settings_for_tmp_path = settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)


def truncate(payload, k):
    """The same payload with each token's ranks cut to `k`, as the parser should cut them."""
    content = payload["choices"][0]["logprobs"]["content"]
    return {"choices": [{"logprobs": {"content": [{"top_logprobs": t["top_logprobs"][:k]} for t in content]}}]}


# --- narrower than k is refused --------------------------------------------------------


@settings_for_tmp_path
@given(payload=narrow_payloads, detector=estimators(n_features=1))
def test_epr_refuses_a_response_narrower_than_k(tmp_path, payload, detector):
    detector = epr(str(write_estimator(tmp_path, "cal.skops", detector)))

    with pytest.raises(ValueError, match=rf"carries {payload_width(payload)} rank\(s\) per token but k={CALIBRATED_K}"):
        detector.predict_proba(payload)


@settings_for_tmp_path
@given(payload=narrow_payloads, weights=dense_weights)
def test_wepr_refuses_a_response_narrower_than_k(tmp_path, payload, weights):
    detector = wepr(str(write_estimator(tmp_path, "w.skops", weights)))

    with pytest.raises(ValueError, match=rf"carries {payload_width(payload)} rank\(s\) per token but k={CALIBRATED_K}"):
        detector.predict_proba(payload)


@settings_for_tmp_path
@given(payload=narrow_payloads, detector=estimators(n_features=1))
def test_the_refusal_names_the_remedy(tmp_path, payload, detector):
    """The caller's fix is upstream, at generation time, so the message has to say so.

    A bare shape error names feature counts and tells the caller nothing about
    `top_logprobs`, which is the knob that actually produced the mismatch.
    """
    detector = epr(str(write_estimator(tmp_path, "cal.skops", detector)))

    with pytest.raises(ValueError, match=f"Regenerate with top_logprobs={CALIBRATED_K}"):
        detector.predict_proba(payload)


@settings_for_tmp_path
@given(payload=narrow_payloads, detector=estimators(n_features=1))
def test_the_refusal_explains_the_direction_of_the_error(tmp_path, payload, detector):
    # which way the score would be wrong is the part that tells a reader whether they can
    # live with it -- a narrow response looks *more* confident, so hallucinations slip past
    detector = epr(str(write_estimator(tmp_path, "cal.skops", detector)))

    with pytest.raises(ValueError, match="more confident than it was"):
        detector.predict_proba(payload)


@settings_for_tmp_path
@given(payload=narrow_payloads, weights=dense_weights)
def test_token_scoring_refuses_a_narrow_response_too(tmp_path, payload, weights):
    # predict_token_proba routes around transform(), so it needs the guard to hold there too
    detector = wepr(str(write_estimator(tmp_path, "w.skops", weights)))

    with pytest.raises(ValueError, match=f"but k={CALIBRATED_K}"):
        detector.predict_token_proba(payload)


# --- a batch's widest member must not hide its narrowest --------------------------------


@settings_for_tmp_path
@given(narrow=narrow_payloads, wide=wide_payloads, detector=estimators(n_features=1), position=st.booleans())
def test_a_narrow_member_is_caught_wherever_it_sits(tmp_path, narrow, wide, detector, position):
    """The whole point of checking per response rather than per batch.

    Padding is per batch, so a wide sibling pads the narrow response up to its own width
    and a batch-level check sees nothing wrong -- while the narrow member is scored on
    ranks it never carried. The offending response is named by position either way.
    """
    detector = epr(str(write_estimator(tmp_path, "cal.skops", detector)))
    batch = [wide, narrow] if position else [narrow, wide]
    index = 1 if position else 0

    with pytest.raises(ValueError, match=rf"Response {index} carries {payload_width(narrow)} rank\(s\)"):
        detector.predict_proba(batch)


# --- wider than k is truncated, not refused ---------------------------------------------


@settings_for_tmp_path
@given(payload=wide_payloads, detector=estimators(n_features=1))
def test_a_response_at_least_k_wide_is_scored(tmp_path, payload, detector):
    scores = epr(str(write_estimator(tmp_path, "cal.skops", detector))).predict_proba(payload)

    assert scores.shape == (1, 2)
    assert np.all((scores >= 0) & (scores <= 1))


@settings_for_tmp_path
@given(payload=wide_payloads, detector=estimators(n_features=1))
def test_epr_drops_ranks_beyond_the_calibrated_k(tmp_path, payload, detector):
    """Truncation has to be exact, or the score drifts with `top_logprobs`.

    Asserted on the features rather than the probabilities: a drawn coefficient can
    saturate the sigmoid, and two saturated probabilities agree whether or not the
    truncation was right.
    """
    front = epr(str(write_estimator(tmp_path, "cal.skops", detector)))[:-1]

    np.testing.assert_allclose(front.transform(payload), front.transform(truncate(payload, CALIBRATED_K)), rtol=1e-6)


@settings_for_tmp_path
@given(payload=wide_payloads, weights=dense_weights)
def test_wepr_drops_ranks_beyond_the_calibrated_k(tmp_path, weights, payload):
    front = wepr(str(write_estimator(tmp_path, "w.skops", weights)))[:-1]

    np.testing.assert_allclose(front.transform(payload), front.transform(truncate(payload, CALIBRATED_K)), rtol=1e-6)


@settings_for_tmp_path
@given(payload=wide_payloads, other=wide_payloads, detector=estimators(n_features=1))
def test_a_score_does_not_move_when_the_response_is_batched(tmp_path, payload, other, detector):
    """Batching must not move a score, now that every member is truncated to the same k.

    The same response scored alone and scored beside a wider one has to come out equal, or
    probabilities stop being comparable between calls.
    """
    detector = epr(str(write_estimator(tmp_path, "cal.skops", detector)))

    alone = float(detector.predict_proba(payload)[0, 1])
    batched = float(detector.predict_proba([payload, other])[0, 1])

    assert alone == pytest.approx(batched, rel=1e-5)


# --- the width the classifier actually receives ------------------------------------------


@settings_for_tmp_path
@given(payload=wide_payloads, weights=dense_weights)
def test_wepr_feature_width_matches_the_calibration(tmp_path, payload, weights):
    # 2k columns (mean branch + max branch) regardless of how wide the response was
    features = wepr(str(write_estimator(tmp_path, "w.skops", weights)))[:-1].transform(payload)

    assert features.shape == (1, 2 * CALIBRATED_K)
