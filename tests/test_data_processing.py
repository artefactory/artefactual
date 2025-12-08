from artefactual.preprocessing.vllm_parser import process_logprobs


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


def test_process_logprobs_empty_input():
    """Test with empty outputs list."""
    assert process_logprobs([], 1) == []


def test_process_logprobs_empty_outputs_list():
    """Test with RequestOutput containing empty outputs."""
    mock_req = MockRequestOutput(outputs=[])
    assert process_logprobs([mock_req], 1) == []


def test_process_logprobs_basic_functionality():
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

    request_output = MockRequestOutput(outputs=[completion_output0, completion_output1])

    # Run function
    result = process_logprobs([request_output], iterations=2)

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


def test_process_logprobs_iterations_parameter():
    """Test that iterations parameter limits the processing."""
    token0_topk = {1: MockLogprob(-1.0)}
    logprobs_seq = [token0_topk]

    comp_out = MockCompletionOutput(logprobs=logprobs_seq)
    # 3 outputs available
    req_out = MockRequestOutput(outputs=[comp_out, comp_out, comp_out])

    # Only ask for 1 iteration
    result = process_logprobs([req_out], iterations=1)
    assert len(result) == 1

    # Ask for 2 iterations
    result = process_logprobs([req_out], iterations=2)
    assert len(result) == 2


def test_process_logprobs_missing_logprobs_in_sequence():
    """Test handling of None or empty logprobs in a sequence."""
    # Case where logprobs is None (if that's possible in vLLM) or empty list
    comp_out_none = MockCompletionOutput(logprobs=None)
    comp_out_empty = MockCompletionOutput(logprobs=[])

    req_out = MockRequestOutput(outputs=[comp_out_none, comp_out_empty])

    result = process_logprobs([req_out], iterations=2)

    assert len(result) == 2
    assert result[0] == {}
    assert result[1] == {}
