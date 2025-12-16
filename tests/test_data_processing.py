from artefactual.preprocessing.vllm_parser import process_vllm_logprobs


# Mocks to simulate vLLM structures
class MockLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class MockCompletionOutput:
    def __init__(self, logprobs):
        self.logprobs = logprobs


class MockRequestOutput:
    def __init__(self, outputs):
        self.outputs = outputs


def test_process_vllm_logprobs_empty_input():
    """Test behavior when an empty sequence of CompletionOutput objects is provided."""
    assert process_vllm_logprobs([]) == []


def test_process_vllm_logprobs_none_logprobs():
    """Test with completion containing None logprobs."""
    # Test with a completion that has None logprobs
    comp_out = MockCompletionOutput(logprobs=None)
    assert process_vllm_logprobs([comp_out]) == [{}]


def test_process_vllm_logprobs_basic_functionality():
    """Test basic functionality with valid logprobs."""
    # Create mock data
    # Sequence 0: 2 tokens
    # Token 0: 2 top candidates
    token0_topk = {101: MockLogprob(-0.1), 102: MockLogprob(-2.5)}
    # Token 1: 1 top candidate
    token1_topk = {201: MockLogprob(-0.5)}

    logprobs_seq0 = [token0_topk, token1_topk]

    # Sequence 1: Empty logprobs (e.g. failed generation or empty)
    logprobs_seq1 = []

    completion_output0 = MockCompletionOutput(logprobs=logprobs_seq0)
    completion_output1 = MockCompletionOutput(logprobs=logprobs_seq1)

    # Run function - now passing completions directly instead of RequestOutput
    result = process_vllm_logprobs([completion_output0, completion_output1])

    # Assertions
    assert len(result) == 2

    # Check sequence 0
    seq0 = result[0]
    assert len(seq0) == 2
    # Note: The order of values in the list depends on dictionary iteration order.
    # In Python 3.7+, insertion order is preserved.
    assert set(seq0[0]) == {-0.1, -2.5}
    assert seq0[1] == [-0.5]

    # Check sequence 1
    seq1 = result[1]
    assert seq1 == {}


def test_process_vllm_logprobs_max_completions_parameter():
    """Test that max_completions parameter limits the processing."""
    token0_topk = {1: MockLogprob(-1.0)}
    logprobs_seq = [token0_topk]

    comp_out = MockCompletionOutput(logprobs=logprobs_seq)
    # 3 completions available
    completions = [comp_out, comp_out, comp_out]

    # Only ask for 1 completion
    result = process_vllm_logprobs(completions, max_completions=1)
    assert len(result) == 1

    # Ask for 2 completions
    result = process_vllm_logprobs(completions, max_completions=2)
    assert len(result) == 2

    # Process all completions (default behavior)
    result = process_vllm_logprobs(completions)
    assert len(result) == 3


def test_process_vllm_logprobs_missing_logprobs_in_sequence():
    """Test handling of None or empty logprobs in a sequence."""
    # Case where logprobs is None (if that's possible in vLLM) or empty list
    comp_out_none = MockCompletionOutput(logprobs=None)
    comp_out_empty = MockCompletionOutput(logprobs=[])

    completions = [comp_out_none, comp_out_empty]

    result = process_vllm_logprobs(completions)

    assert len(result) == 2
    assert result[0] == {}
    assert result[1] == {}


def test_process_vllm_logprobs_no_index_error_with_large_max():
    """Test that requesting more completions than available doesn't raise IndexError."""
    token0_topk = {1: MockLogprob(-1.0)}
    logprobs_seq = [token0_topk]
    comp_out = MockCompletionOutput(logprobs=logprobs_seq)

    # Only 2 completions available
    completions = [comp_out, comp_out]

    # Request more than available - should not raise IndexError
    result = process_vllm_logprobs(completions, max_completions=10)

    # Should only return what's available
    assert len(result) == 2


def test_process_vllm_logprobs_processes_all_by_default():
    """Test that all completions are processed when max_completions is not specified."""
    token0_topk = {1: MockLogprob(-1.0)}
    logprobs_seq = [token0_topk]
    comp_out = MockCompletionOutput(logprobs=logprobs_seq)

    # 5 completions available
    completions = [comp_out] * 5

    # Should process all completions when max_completions is not specified
    result = process_vllm_logprobs(completions)
    assert len(result) == 5


def test_process_vllm_logprobs_batched_results_not_dropped():
    """Test that batched request outputs are not silently ignored."""
    # Create 3 different completions with different logprobs to verify they're all processed
    token0_topk_1 = {1: MockLogprob(-1.0)}
    token0_topk_2 = {2: MockLogprob(-2.0)}
    token0_topk_3 = {3: MockLogprob(-3.0)}

    comp_out1 = MockCompletionOutput(logprobs=[token0_topk_1])
    comp_out2 = MockCompletionOutput(logprobs=[token0_topk_2])
    comp_out3 = MockCompletionOutput(logprobs=[token0_topk_3])

    completions = [comp_out1, comp_out2, comp_out3]

    result = process_vllm_logprobs(completions)

    # All three completions should be processed
    assert len(result) == 3
    # Verify each has unique data
    assert result[0][0] == [-1.0]
    assert result[1][0] == [-2.0]
    assert result[2][0] == [-3.0]
