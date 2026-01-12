"""
Module for parsing model outputs from various sources to extract log probabilities.
Each format is handled by a dedicated parser function, defined in their respective modules.
"""

from typing import Any

from artefactual.preprocessing import (
    is_openai_responses_api,
    process_openai_chat_completion,
    process_openai_responses_api,
    process_vllm_logprobs,
)


def parse_model_outputs(outputs: Any) -> list[dict[int, list[float]]]:
    """
    Parse different output formats to extract logprobs.

    Args:
        outputs: Model outputs. Can be:
                    - List of vLLM RequestOutput objects.
                    - OpenAI ChatCompletion object (or dict).
                    - OpenAI Responses object (or dict).

    Returns:
        List of dictionaries mapping token indices to lists of log probs.

    Raises:
        TypeError: If the output format is not supported.
    """
    # vLLM parser
    if isinstance(outputs, list) and len(outputs) > 0 and hasattr(outputs[0], "outputs"):
        if not outputs[0].outputs:
            return []
        iterations = len(outputs[0].outputs)
        return process_vllm_logprobs(outputs, iterations)

    # OpenAI parser for classic ChatCompletion
    if hasattr(outputs, "choices") or (isinstance(outputs, dict) and "choices" in outputs):
        choices = outputs.choices if hasattr(outputs, "choices") else outputs["choices"]
        return process_openai_chat_completion(outputs, iterations=len(choices))

    # OpenAI parser for Responses API
    if is_openai_responses_api(outputs):
        return process_openai_responses_api(outputs)

    msg = (
        f"Unsupported output format: {type(outputs).__name__}. "
        "Expected vLLM RequestOutput, OpenAI ChatCompletion, or OpenAI Responses object."
    )
    raise TypeError(msg)
