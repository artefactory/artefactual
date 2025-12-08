"""Tests for entropic transformers and token-scoring estimators."""

import numpy as np
from absl.testing import absltest
from beartype.roar import BeartypeCallHintParamViolation
from sklearn.exceptions import NotFittedError

from artefactual.estimators import TokenScoringLogisticRegression
from artefactual.parsers import OpenAIParsedCompletionV1, parsed_top_logprobs_tensor
from artefactual.transformers import EntropicContributionTransformer, SequenceStatAggregator


class TransformersTest(absltest.TestCase):
    """Tests for entropic transformers."""

    def test_entropic_contribution_transformer_shape(self):
        """Transformer returns same shape and zeros padding stays zero."""
        x = np.array(
            [
                [[-1.0, -1.0], [-2.0, -2.0], [0.0, 0.0]],
                [[-0.5, -0.5], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        transformer = EntropicContributionTransformer(k=2, padding_value=0.0)
        seq = transformer.transform(x)
        self.assertEqual(seq.shape, x.shape)
        # Padding rows should remain zeroed
        self.assertTrue(np.allclose(seq[0, 2], 0.0))
        self.assertTrue(np.allclose(seq[1, 1:], 0.0))

    def test_sequence_stat_aggregator_mean_respects_padding(self):
        """Mean aggregation divides by valid tokens only."""
        seq = np.array(
            [
                [[1.0, 3.0], [2.0, 2.0], [0.0, 0.0]],
            ]
        )
        agg = SequenceStatAggregator(stat="mean", k=2)
        features = agg.transform(seq)
        # Two valid tokens -> mean over first two rows
        expected = np.array([[1.5, 2.5]])
        self.assertTrue(np.allclose(features, expected))

    def test_entropic_contribution_with_padding_value(self):
        """Non-zero padding stays zeroed and entropy math is correct."""
        x = np.array(
            [
                [[np.log(0.5), np.log(0.25)], [-99.0, -99.0]],
            ]
        )
        transformer = EntropicContributionTransformer(k=2, padding_value=-99.0)
        s = transformer.transform(x)
        # Entropy for p=0.5 and p=0.25 are both 0.5
        self.assertTrue(np.allclose(s[0, 0], [0.5, 0.5], atol=1e-6))
        self.assertTrue(np.allclose(s[0, 1], [0.0, 0.0], atol=1e-6))
        names = transformer.get_feature_names_out()
        self.assertListEqual(names.tolist(), ["entropy_rank_0", "entropy_rank_1"])

    def test_sequence_stat_aggregator_min_ignores_padding(self):
        """Padding rows should not affect min aggregation."""
        seq = np.array([[[1.0, 2.0], [0.0, 0.0]]])
        agg = SequenceStatAggregator(stat="min", k=2)
        features = agg.transform(seq)
        self.assertTrue(np.allclose(features, [[1.0, 2.0]]))
        names = agg.get_feature_names_out()
        self.assertListEqual(names.tolist(), ["min_rank_0", "min_rank_1"])

    def test_sequence_stat_invalid_stat_raises(self):
        with self.assertRaises(BeartypeCallHintParamViolation):
            SequenceStatAggregator(stat="median", k=1)


class TokenScoringMixinTest(absltest.TestCase):
    """Tests for token scoring mixin."""

    def test_token_scores_with_zero_weights(self):
        """Flat weights should yield 0.5 probabilities and harmonic 0.5."""
        clf = TokenScoringLogisticRegression(k=2)
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.zeros((1, 4))
        clf.intercept_ = np.array([0.0])
        clf.n_features_in_ = 4  # type: ignore[attr-defined]

        arr = np.zeros((1, 2, 2))
        scores = clf.transform(arr)
        self.assertEqual(scores.shape, (1, 2))
        self.assertTrue(np.allclose(scores, 0.5))

    def test_transform_requires_fitted_estimator(self):
        clf = TokenScoringLogisticRegression(k=2)
        with self.assertRaises(NotFittedError):
            clf.transform(np.zeros((1, 1, 2)))

    def test_transform_validates_dimensions(self):
        clf = TokenScoringLogisticRegression(k=2)
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.zeros((1, 4))
        clf.intercept_ = np.array([0.0])
        clf.n_features_in_ = 4  # type: ignore[attr-defined]
        with self.assertRaises(BeartypeCallHintParamViolation):
            clf.transform(np.zeros((2, 2)))

    def test_transform_validates_feature_blocks(self):
        clf = TokenScoringLogisticRegression(k=3)
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.zeros((1, 5))
        clf.intercept_ = np.array([0.0])
        clf.n_features_in_ = 5  # type: ignore[attr-defined]
        with self.assertRaisesRegex(ValueError, "not divisible by K=3"):
            clf.transform(np.zeros((1, 1, 3)))

    def test_transform_validates_component_weights_length(self):
        clf = TokenScoringLogisticRegression(k=2, component_weights=[1.0])
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.zeros((1, 4))
        clf.intercept_ = np.array([0.0])
        clf.n_features_in_ = 4  # type: ignore[attr-defined]
        with self.assertRaisesRegex(ValueError, "component weights"):
            clf.transform(np.zeros((1, 1, 2)))

    def test_harmonic_mean_respects_component_weights(self):
        """Weighted harmonic mean should favor larger weights."""
        clf = TokenScoringLogisticRegression(k=1, component_weights=[0.25, 0.75])
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.array([[1.0, -1.0]])
        clf.intercept_ = np.array([0.0])
        clf.n_features_in_ = 2  # type: ignore[attr-defined]

        arr = np.array([[[1.0]]])
        scores = clf.transform(arr)

        block_coefs = clf.coef_[0].reshape(2, 1)
        logits_blocks = np.dot(arr, block_coefs.T) + clf.intercept_[0]
        probs_blocks = 1 / (1 + np.exp(-logits_blocks))
        weights = np.array(clf.component_weights).reshape(1, 1, 2)
        weighted_inverse_sum = np.sum(weights / (probs_blocks + 1e-9), axis=2)
        expected = np.sum(weights) / weighted_inverse_sum

        self.assertTrue(np.allclose(scores, expected))


class FeatureBuilderTest(absltest.TestCase):
    """Tests for feature-building helpers."""

    def test_build_token_features_shapes_and_padding(self):
        parsed = [
            OpenAIParsedCompletionV1.from_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "Hi",
                                        "logprobs": {
                                            "tokens": ["Hi"],
                                            "token_logprobs": [np.log(0.5)],
                                            "top_logprobs": [{"Hi": np.log(0.5)}],
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        ]
        tensor = parsed_top_logprobs_tensor(parsed, top_k=2, max_tokens=3, padding_value=0.0)  # type: ignore[arg-type]
        feats = EntropicContributionTransformer(k=2, padding_value=0.0).transform(tensor)
        self.assertEqual(feats.shape, tensor.shape)
        # Padding tokens remain zero
        self.assertTrue(np.allclose(feats[0, 1:], 0.0))

    def test_build_sequence_features_concatenates_stats(self):
        parsed = [
            OpenAIParsedCompletionV1.from_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": "A",
                                        "logprobs": {
                                            "tokens": ["A"],
                                            "token_logprobs": [np.log(0.5)],
                                            "top_logprobs": [{"A": np.log(0.5)}],
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )
        ]
        tensor = parsed_top_logprobs_tensor(parsed, top_k=1, max_tokens=2, padding_value=0.0)  # type: ignore[arg-type]
        token_feats = EntropicContributionTransformer(k=1, padding_value=0.0).transform(tensor)
        feats = np.concatenate(
            [
                SequenceStatAggregator(stat="mean", k=1).transform(token_feats),
                SequenceStatAggregator(stat="max", k=1).transform(token_feats),
            ],
            axis=1,
        )
        self.assertEqual(feats.shape, (1, 2))  # two stats concatenated
        self.assertTrue(np.allclose(feats[0, 0], feats[0, 1]))
        self.assertAlmostEqual(feats[0, 0], 0.5, places=3)


if __name__ == "__main__":
    absltest.main()
