import os
import sys

from vllm import LLM, SamplingParams

# Add src to path so we can import artefactual if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from artefactual.scoring.entropy_methods.epr import EPR
from artefactual.scoring.entropy_methods.wepr import WEPR


def vllm_example():
    # ==========================================
    # OPTION 1: vLLM Example
    # ==========================================

    # # Initialize vLLM
    llm = LLM(model="mistralai/Ministral-8B-Instruct-2410")
    prompts = ["The capital of France is"]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, logprobs=15, max_tokens=20)

    print("Running inference with vLLM...")  # noqa: T201
    outputs = llm.generate(prompts, sampling_params)

    # Compute EPR
    epr = EPR()
    scores = epr.compute(outputs)
    print(f"EPR Scores: {scores}")  # noqa: T201

    # Detailed scores (per token)
    seq_scores = epr.compute(outputs)
    token_scores = epr.compute_token_scores(outputs)
    print(f"Sequence Scores: {seq_scores}")  # noqa: T201
    print(f"Token Scores (first seq): {token_scores[0]}")  # noqa: T201

    # Compute WEPR
    wepr = WEPR(model="mistralai/Ministral-8B-Instruct-2410")
    wepr.compute(outputs)


def openai_example():
    # ==========================================
    # OPTION 2: OpenAI Example (Mocked)
    # ==========================================

    # Mock response mimicking OpenAI Responses API structure

    mock_response = {
        "id": "resp_simulation",
        "object": "response",
        "created": 1234567890,
        "model": "gpt-4-simulation",
        "output": [
            {
                "index": 0,
                "content": [
                    {
                        "type": "text",
                        "text": "Paris",
                        "logprobs": [
                            {
                                "token": "Par",
                                "logprob": -0.001,
                                "top_logprobs": [
                                    {"token": "Par", "logprob": -0.001},
                                    {"token": "Lon", "logprob": -7.5},
                                    {"token": "Ber", "logprob": -9.2},
                                ],
                            },
                            {
                                "token": "is",
                                "logprob": -0.002,
                                "top_logprobs": [
                                    {"token": "is", "logprob": -0.002},
                                    {"token": "don", "logprob": -8.0},
                                    {"token": "ma", "logprob": -8.5},
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }

    print("\nRunning EPR on Mock OpenAI Response...")  # noqa: T201
    epr = EPR()
    scores = epr.compute(mock_response)
    print(f"EPR Score: {scores[0]}")  # noqa: T201

    wepr = WEPR(model="mistralai/Ministral-8B-Instruct-2410")
    wepr.compute(mock_response)


if __name__ == "__main__":
    vllm_example()
    # openai_example()
