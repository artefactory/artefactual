import json
from pathlib import Path

from beartype.door import is_bearable
from hypothesis import strategies as st

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


# --- calibration and weight files ------------------------------------------------------

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
def epr_calibration(draw):
    """An EPR calibration file: an intercept plus a single `mean_entropy` coefficient."""
    return {
        "intercept": draw(coefficient_values),
        "coefficients": {"mean_entropy": draw(coefficient_values)},
    }


def write_json(directory, name, payload):
    """Write `payload` to `directory/name` and return the path, as callers pass paths around."""
    path = Path(directory) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
