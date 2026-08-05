"""Parsing tests for the OpenAI wire formats.

Payloads come from the strategies in `tests/conftest.py`, which build the response models
directly. Each generated payload is exercised in both representations a caller may hold:
the model instance (attribute access, as provider SDKs return) and `model_dump()` (the
plain dict you get from JSON).
"""

import numpy as np
import pytest
from conftest import chat_completions, expected_sequence_count, openai_payloads, responses_api
from hypothesis import given, settings
from hypothesis import strategies as st

from artefactual.preprocessing.parser import LogProbParser, parse_sampled_token_logprobs, parse_top_logprobs


@given(payload=chat_completions)
def test_chat_completion_yields_one_sequence_per_choice(payload):
    assert len(parse_top_logprobs(payload)) == len(payload.choices)


@given(payload=responses_api)
def test_responses_api_yields_one_sequence_per_output_item(payload):
    assert len(parse_top_logprobs(payload)) == len(payload.output)


@given(payload=openai_payloads)
def test_dict_and_object_payloads_parse_identically(payload):
    # the models normalise both representations, so callers can pass either
    assert parse_top_logprobs(payload.model_dump()) == parse_top_logprobs(payload)


@given(payload=openai_payloads)
def test_logprobs_are_sorted_descending(payload):
    # the parser promises highest-probability rank first; entropy contributions depend on it
    for sequence in parse_top_logprobs(payload):
        for ranks in sequence.values():
            assert ranks == sorted(ranks, reverse=True)


@given(payload=openai_payloads)
def test_parsed_logprobs_stay_non_positive_and_finite(payload):
    for sequence in parse_top_logprobs(payload):
        for ranks in sequence.values():
            assert all(np.isfinite(value) and value <= 0 for value in ranks)


@given(batch=st.lists(openai_payloads, min_size=1, max_size=5))
def test_batch_is_the_concatenation_of_its_parts(batch):
    # a calibration dataset is one API call per example; the batch must not lose sequences
    assert len(parse_top_logprobs(batch)) == sum(expected_sequence_count(p) for p in batch)


@given(batch=st.lists(openai_payloads, min_size=2, max_size=4))
def test_batches_mixing_wire_formats_and_representations_parse(batch):
    mixed = [p if index % 2 else p.model_dump() for index, p in enumerate(batch)]
    assert len(parse_top_logprobs(mixed)) == sum(expected_sequence_count(p) for p in batch)


@given(batch=st.lists(openai_payloads, min_size=1, max_size=4))
@settings(deadline=None)
def test_transform_pads_ragged_batches_to_a_dense_cube(batch):
    parsed = parse_top_logprobs(batch)
    array = LogProbParser().transform(batch)

    expected_tokens = max((max(sequence.keys()) + 1 for sequence in parsed if sequence), default=0)
    expected_ranks = max((len(ranks) for sequence in parsed for ranks in sequence.values()), default=0)
    assert array.shape == (len(parsed), expected_tokens, expected_ranks)


@given(payload=openai_payloads)
def test_sampled_logprobs_yield_one_array_per_sequence(payload):
    sampled = parse_sampled_token_logprobs(payload)
    assert len(sampled) == expected_sequence_count(payload)
    assert all(array.ndim == 1 for array in sampled)


@given(payload=openai_payloads)
def test_sampled_logprobs_stay_non_positive_and_finite(payload):
    for array in parse_sampled_token_logprobs(payload):
        assert np.isfinite(array).all()
        assert (array <= 0).all()


@given(payload=openai_payloads)
def test_sampled_logprobs_match_across_representations(payload):
    from_object = parse_sampled_token_logprobs(payload)
    from_mapping = parse_sampled_token_logprobs(payload.model_dump())
    assert [a.tolist() for a in from_object] == [a.tolist() for a in from_mapping]


def test_empty_batch_returns_no_sequences():
    assert parse_top_logprobs([]) == []
    assert LogProbParser().transform([]).shape == (0, 0, 0)


def test_unsupported_payload_raises():
    with pytest.raises(TypeError, match="Unsupported output format"):
        parse_top_logprobs("not-a-response")
