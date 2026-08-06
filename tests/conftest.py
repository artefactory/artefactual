import json
from pathlib import Path

import numpy as np
import skops.io as sio
from beartype.door import is_bearable
from hypothesis import strategies as st
from sklearn.linear_model import LogisticRegression

from artefactual.preprocessing.response_models import (
    ChatChoice,
    ChatChoiceLogprobs,
    ChatCompletion,
    ResponseContentPart,
    ResponseOutputItem,
    ResponsesPayload,
    TokenLogprobs,
    TopLogprob,
)

# Payloads are built from the response models themselves, so a change to the models
# changes what the tests generate — the shapes cannot drift apart.

# logprobs are <= 0 and finite; the parser rejects anything else
logprob_values = st.floats(min_value=-25.0, max_value=0.0, allow_nan=False, allow_infinity=False)

top_logprobs = st.lists(st.builds(TopLogprob, logprob=logprob_values), min_size=1, max_size=8)

token_logprobs = st.builds(
    lambda ranks: TokenLogprobs(logprob=ranks[0].logprob, top_logprobs=ranks),
    top_logprobs,
)

# ragged token counts within a sequence
token_sequences = st.lists(token_logprobs, min_size=1, max_size=6)

chat_completions = st.builds(
    ChatCompletion,
    choices=st.lists(
        st.builds(ChatChoice, logprobs=st.builds(ChatChoiceLogprobs, content=token_sequences)),
        min_size=1,
        max_size=4,
    ),
)

responses_api = st.builds(
    ResponsesPayload,
    output=st.lists(
        st.builds(
            ResponseOutputItem,
            content=st.lists(st.builds(ResponseContentPart, logprobs=token_sequences), min_size=1, max_size=1),
        ),
        min_size=1,
        max_size=4,
    ),
)

openai_payloads = st.one_of(chat_completions, responses_api)


def expected_sequence_count(payload):
    """Sequences the payload should yield: one per choice, or one per output item."""
    return len(payload.choices) if is_bearable(payload, ChatCompletion) else len(payload.output)


# --- parsed logprobs -------------------------------------------------------------------
# The shape the scorers consume: one dict per sequence, token position -> descending ranks.
# Built directly rather than by round-tripping a payload so a parser bug cannot mask a
# scorer bug (and vice versa).


@st.composite
def rank_vectors(draw, min_ranks=1, max_ranks=20):
    """One token's top-k logprobs, descending."""
    ranks = draw(st.lists(logprob_values, min_size=min_ranks, max_size=max_ranks))
    return sorted(ranks, reverse=True)


@st.composite
def parsed_sequences(draw, min_sequences=1, max_sequences=4, min_ranks=1, max_ranks=20):
    """A batch of parsed sequences with a rank width that is constant within a sequence.

    Real top-k responses hold k fixed for a request, and `np.asarray` on a ragged
    list would build an object array, so the width is drawn once per sequence.
    """
    sequences = []
    for _ in range(draw(st.integers(min_sequences, max_sequences))):
        width = draw(st.integers(min_ranks, max_ranks))
        n_tokens = draw(st.integers(1, 6))
        sequences.append({
            position: draw(rank_vectors(min_ranks=width, max_ranks=width)) for position in range(n_tokens)
        })
    return sequences


@st.composite
def chat_payloads_of_fixed_width(draw, min_ranks=1, max_ranks=20):
    """A raw ChatCompletion mapping whose every token carries the same top-k width.

    Emitted as a plain mapping rather than a model instance because that is what a caller
    hands the pipeline, and the width is what these tests vary. Recover it with
    `payload_width`.
    """
    width = draw(st.integers(min_ranks, max_ranks))
    content = [
        {"top_logprobs": [{"logprob": value} for value in draw(rank_vectors(min_ranks=width, max_ranks=width))]}
        for _ in range(draw(st.integers(1, 4)))
    ]
    return {"choices": [{"logprobs": {"content": content}}]}


def payload_width(payload):
    """The top-k width of a payload from `chat_payloads_of_fixed_width`."""
    return len(payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"])


@st.composite
def logprob_cubes(draw, min_sequences=1, max_sequences=3, min_tokens=1, max_tokens=4, min_ranks=2, max_ranks=8):
    """The `(n_sequences, n_tokens, k)` array `LogProbParser` hands the scorers.

    Rectangular and descending along the rank axis, which is what the parser guarantees:
    it sizes the rank axis once for the batch and sorts each token's ranks.
    """
    tokens = draw(st.integers(min_tokens, max_tokens))
    width = draw(st.integers(min_ranks, max_ranks))
    return np.array([
        [draw(rank_vectors(min_ranks=width, max_ranks=width)) for _ in range(tokens)]
        for _ in range(draw(st.integers(min_sequences, max_sequences)))
    ])


# --- detector and weight files ------------------------------------------------------

coefficient_values = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


@st.composite
def wepr_weights(draw, min_k=1, max_k=15):
    """A WEPR weights file: an intercept plus dense `mean_rank_i` / `max_rank_i` pairs."""
    k = draw(st.integers(min_k, max_k))
    coefficients = {}
    for rank in range(1, k + 1):
        coefficients[f"mean_rank_{rank}"] = draw(coefficient_values)
        coefficients[f"max_rank_{rank}"] = draw(coefficient_values)
    return {"intercept": draw(coefficient_values), "coefficients": coefficients}


@st.composite
def epr_detector(draw):
    """An EPR detector file: an intercept plus a single `mean_entropy` coefficient."""
    return {
        "intercept": draw(coefficient_values),
        "coefficients": {"mean_entropy": draw(coefficient_values)},
    }


def write_json(directory, name, payload):
    """Write `payload` to `directory/name` and return the path, as callers pass paths around."""
    path = Path(directory) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- estimators on disk --------------------------------------------------------------
#
# Detectors are built and written here rather than fetched, so every test names a path.
# That is the whole reason these tests stay offline: `resolve_estimator` returns a local
# file without consulting the Hub, so no test crosses the network seam and none has to be
# skipped, mocked, or given credentials for a private repository.


def fitted_logistic(intercept, coefficients):
    """A `LogisticRegression` presented as fitted, from an intercept and one coefficient row.

    Assigns the attributes `fit` would set. Detectors are coefficients that were fit
    elsewhere, so there is no training data here to fit from.
    """
    estimator = LogisticRegression()
    estimator.coef_ = np.array([coefficients], dtype=np.float64)
    estimator.intercept_ = np.array([intercept], dtype=np.float64)
    estimator.classes_ = np.array([0, 1])
    estimator.n_features_in_ = len(coefficients)
    estimator.n_iter_ = np.array([0])
    return estimator


def write_estimator(directory, name, estimator):
    """Write `estimator` to `directory/name` as skops and return the path."""
    path = Path(directory) / name
    sio.dump(estimator, path)
    return path


@st.composite
def estimators(draw, n_features=None):
    """A detector: an intercept and one coefficient per feature.

    One strategy for both reductions, because they produce the same object and are loaded
    by the same code -- EPR is one pooled coefficient, WEPR is `2k` per-rank ones, and
    nothing else about them differs. Pass `n_features` when the width is what the test is
    about; leave it out when the test holds for any width.
    """
    width = n_features if n_features is not None else draw(st.integers(1, 30))
    return fitted_logistic(draw(coefficient_values), [draw(coefficient_values) for _ in range(width)])
