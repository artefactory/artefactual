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
