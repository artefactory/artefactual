import numpy as np

from artefactual.scoring.probability_methods.sentence_proba import SentenceProbabilityScorer


def test_sentence_probability_scorer_compute():
    scorer = SentenceProbabilityScorer()

    # Test with list of 1D arrays (multiple sentences of different lengths)
    inputs = [np.array([-0.1, -0.2, -0.3]), np.array([-0.4, -0.5])]
    result = scorer.compute(inputs)

    expected_sums = [-0.6, -0.9]
    expected_probs = np.exp(expected_sums).tolist()

    assert np.allclose(result, expected_probs)


def test_sentence_probability_scorer_compute_2d():
    scorer = SentenceProbabilityScorer()

    # Test with list of 1D arrays (multiple sentences of same length)
    inputs = [np.array([-0.1, -0.2]), np.array([-0.3, -0.4])]
    result = scorer.compute(inputs)

    expected_sums = [-0.3, -0.7]
    expected_probs = np.exp(expected_sums).tolist()

    assert np.allclose(result, expected_probs)


def test_sentence_probability_scorer_compute_empty_batch():
    scorer = SentenceProbabilityScorer()
    assert scorer.compute([]) == []


def test_sentence_probability_scorer_compute_empty_sequence_returns_zero():
    scorer = SentenceProbabilityScorer()
    result = scorer.compute([np.array([])])
    assert result == [0.0]


def test_sentence_probability_scorer_compute_mixed_empty_and_non_empty():
    scorer = SentenceProbabilityScorer()
    inputs = [np.array([]), np.array([-0.1, -0.2])]
    result = scorer.compute(inputs)
    expected_probs = [0.0, float(np.exp(-0.3))]
    assert np.allclose(result, expected_probs)


def test_sentence_probability_scorer_compute_token_scores():
    scorer = SentenceProbabilityScorer()

    inputs = [np.array([-0.1, -0.2, -0.3]), np.array([-0.4, -0.5])]
    result = scorer.compute_token_scores(inputs)

    expected_probs = [np.exp(seq) for seq in inputs]

    for r, e in zip(result, expected_probs, strict=False):
        np.testing.assert_array_almost_equal(r, e)


def test_sentence_probability_scorer_compute_token_scores_2d():
    scorer = SentenceProbabilityScorer()

    inputs = [np.array([-0.1, -0.2]), np.array([-0.3, -0.4])]
    result = scorer.compute_token_scores(inputs)

    expected_probs = [np.exp(seq) for seq in inputs]

    for r, e in zip(result, expected_probs, strict=False):
        np.testing.assert_array_almost_equal(r, e)
