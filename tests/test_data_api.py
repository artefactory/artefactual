"""Tests for data module API exposure."""

import artefactual.data


def test_data_model_exposed():
    """Test that data model classes are properly exposed in the data module."""
    # Check that the classes are in __all__
    assert "TokenLogprob" in artefactual.data.__all__
    assert "Completion" in artefactual.data.__all__
    assert "Result" in artefactual.data.__all__
    assert "Dataset" in artefactual.data.__all__
    
    # Check that they can be imported
    from artefactual.data import Completion, Dataset, Result, TokenLogprob
    
    assert Completion is not None
    assert Dataset is not None
    assert Result is not None
    assert TokenLogprob is not None


def test_data_model_classes_work():
    """Test that the data model classes can be instantiated."""
    from artefactual.data import Completion, Dataset, Result, TokenLogprob
    
    # Test TokenLogprob
    token_logprob = TokenLogprob(token="test", logprob=-0.5, rank=1)
    assert token_logprob.token == "test"
    assert token_logprob.logprob == -0.5
    assert token_logprob.rank == 1
    
    # Test Completion
    completion = Completion(token_logprobs={0: [0.1, 0.2]})
    assert completion.token_logprobs == {0: [0.1, 0.2]}
    
    # Test Result
    result = Result(
        query_id="q1",
        query="What is the capital of France?",
        expected_answers=["Paris"],
        generated_answers=[{"answer": "Paris"}],
        token_logprobs=[[[0.1, 0.2]]]
    )
    assert result.query_id == "q1"
    assert result.query == "What is the capital of France?"
    
    # Test Dataset
    dataset = Dataset(results=[result])
    assert len(dataset.results) == 1
    assert dataset.results[0].query_id == "q1"
