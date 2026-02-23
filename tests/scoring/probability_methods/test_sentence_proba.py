import numpy as np

from artefactual.scoring.probability_methods.sentence_proba import SentenceProbabilityScorer


def test_sentence_probability_scorer_compute():
    scorer = SentenceProbabilityScorer()

    # Test with 1D array of lists (multiple sentences of different lengths)
    inputs = np.array([[-0.1, -0.2, -0.3], [-0.4, -0.5]], dtype=object)
    result = scorer.compute(inputs)

    expected_sums = [-0.6, -0.9]
    expected_probs = np.exp(expected_sums).tolist()

    assert np.allclose(result, expected_probs)


def test_sentence_probability_scorer_compute_2d():
    scorer = SentenceProbabilityScorer()

    # Test with 2D array (multiple sentences of same length)
    inputs = np.array([[-0.1, -0.2], [-0.3, -0.4]])
    result = scorer.compute(inputs)

    expected_sums = [-0.3, -0.7]
    expected_probs = np.exp(expected_sums).tolist()

    assert np.allclose(result, expected_probs)


def test_sentence_probability_scorer_compute_token_scores():
    scorer = SentenceProbabilityScorer()

    inputs = np.array([[-0.1, -0.2, -0.3], [-0.4, -0.5]], dtype=object)
    result = scorer.compute_token_scores(inputs)

    expected_probs = [np.exp(seq) for seq in inputs]

    for r, e in zip(result, expected_probs, strict=False):
        np.testing.assert_array_almost_equal(r, e)


def test_sentence_probability_scorer_compute_token_scores_2d():
    scorer = SentenceProbabilityScorer()

    inputs = np.array([[-0.1, -0.2], [-0.3, -0.4]])
    result = scorer.compute_token_scores(inputs)

    expected_probs = [np.exp(seq) for seq in inputs]

    for r, e in zip(result, expected_probs, strict=False):
        np.testing.assert_array_almost_equal(r, e)
