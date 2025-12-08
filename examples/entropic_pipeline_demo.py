"""Demo script for entropic hallucination scoring pipeline."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from artefactual.estimators import TokenScoringLogisticRegression
from artefactual.parsers import OpenAIParsedCompletionV1, parsed_top_logprobs_tensor
from artefactual.transformers import EntropicContributionTransformer, SequenceStatAggregator


def _mock_openai_response(seq_len: int, k: int, rng: np.random.RandomState):
    """Create a minimal OpenAI-style completion payload with logprobs."""
    # Dirichlet to make probs sum to 1; convert to log space
    probs = rng.dirichlet(alpha=np.ones(k), size=seq_len)
    logprobs = np.log(probs)
    tokens = [f"tok_{i}" for i in range(seq_len)]
    vocab_tokens = [f"tok_{j}" for j in range(k)]
    top_logprobs = [{vocab_tokens[j]: logprobs[i, j] for j in range(k)} for i in range(seq_len)]
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "".join(tokens),
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": logprobs[:, 0].tolist(),
                                "top_logprobs": top_logprobs,
                            },
                        }
                    ],
                },
            }
        ]
    }


def generate_mock_data(n: int = 200, k: int = 15, max_len: int = 30):
    """Generate mock OpenAI responses and labels."""
    rng = np.random.RandomState(42)
    responses = []
    labels = rng.randint(0, 2, size=n)
    for _ in labels:
        seq_len = rng.randint(5, max_len)
        response = _mock_openai_response(seq_len, k, rng)
        responses.append(OpenAIParsedCompletionV1.from_response(response))
    return responses, labels


def build_pipeline(k: int = 15):
    """Construct the FeatureUnion + classifier pipeline."""
    feature_union = FeatureUnion(
        [
            ("mean_stat", SequenceStatAggregator(stat="mean", k=k)),
            ("max_stat", SequenceStatAggregator(stat="max", k=k)),
        ]
    )

    return Pipeline(
        [
            ("entropic_S", EntropicContributionTransformer(k=k)),
            ("features", feature_union),
            ("clf", TokenScoringLogisticRegression(k=k)),
        ]
    )


def run_demo():
    """Run a lightweight GridSearchCV and print sample outputs."""
    parsed, labels = generate_mock_data()
    x_train, x_test, y_train, _ = train_test_split(parsed, labels, random_state=42)

    # Precompute tensors so the pipeline can operate on arrays
    train_token_features = parsed_top_logprobs_tensor(x_train, top_k=15)
    test_token_features = parsed_top_logprobs_tensor(x_test, top_k=15)

    grid = GridSearchCV(
        build_pipeline(),
        {"clf__component_weights": [(1.0, 1.0), (1.0, 3.0)]},
        cv=2,
        scoring="accuracy",
    )
    grid.fit(train_token_features, y_train)

    best_pipe = grid.best_estimator_
    logging.info("Best Params: %s", grid.best_params_)
    logging.info(
        "Feature Names Sample: %s ... %s",
        best_pipe.named_steps["features"].get_feature_names_out()[:3],
        best_pipe.named_steps["features"].get_feature_names_out()[-3:],
    )

    x_test_s = best_pipe.named_steps["entropic_S"].transform(test_token_features)
    token_scores = best_pipe.named_steps["clf"].transform(x_test_s)
    is_valid = np.any(x_test_s != 0.0, axis=2)
    token_scores[~is_valid] = -1.0

    logging.info("Token scores shape: %s", token_scores.shape)
    logging.info("First sequence valid token scores: %s", token_scores[0][token_scores[0] != -1.0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_demo()
