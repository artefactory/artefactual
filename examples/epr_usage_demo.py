import sys
from pathlib import Path

from vllm import LLM, SamplingParams

from artefactual.preprocessing import parse_model_outputs
from artefactual.scoring import EPR, WEPR

# Add src to path so we can import artefactual if not installed
_current_dir = Path(__file__).resolve().parent
sys.path.append(str(_current_dir.parent / "src"))


def vllm_example() -> None:
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
    epr = EPR()  # initialize with default calibration
    parsed_logprobs = parse_model_outputs(outputs)  # parse the outputs to get logprobs

    scores = epr.compute(parsed_logprobs)
    print(f"EPR Scores: {scores}")  # noqa: T201

    # Detailed scores (per token)
    seq_scores = epr.compute(parsed_logprobs)
    token_scores = epr.compute_token_scores(parsed_logprobs)
    print(f"Sequence Scores: {seq_scores}")  # noqa: T201
    print(f"Token Scores (first seq): {token_scores[0]}")  # noqa: T201

    # Compute WEPR
    weights_path = _current_dir.parent / "src" / "artefactual" / "data" / "weights_ministral.json"
    wepr = WEPR(pretrained_model_name_or_path=str(weights_path))
    wepr.compute(parsed_logprobs)


def openai_example() -> None:
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
    mock_logprobs = parse_model_outputs(mock_response)
    scores = epr.compute(mock_logprobs)  # parse the outputs to get logprobs
    print(f"EPR Score: {scores[0]}")  # noqa: T201

    weights_path = _current_dir.parent / "src" / "artefactual" / "data" / "weights_ministral.json"
    wepr = WEPR(pretrained_model_name_or_path=str(weights_path))
    wepr.compute(mock_logprobs)


if __name__ == "__main__":
    vllm_example()
    # openai_example()
