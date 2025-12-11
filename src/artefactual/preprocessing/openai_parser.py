from typing import Any


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to safely get a value from an object attribute or dict key."""
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _extract_logprobs_from_token(token_data: Any) -> list[float]:
    """Extracts logprobs from a single token's data."""
    top_logprobs = _get_val(token_data, "top_logprobs", [])

    if not top_logprobs:
        return []

    probs = []
    for item in top_logprobs:
        val = _get_val(item, "logprob")
        if val is not None:
            probs.append(float(val))
    return probs


def _process_single_choice(choice: Any) -> dict[int, list[float]]:
    """Processes logprobs for a single choice object."""
    seq = {}

    logprobs_obj = _get_val(choice, "logprobs")
    if not logprobs_obj:
        return seq

    content_logprobs = _get_val(logprobs_obj, "content", [])
    if not content_logprobs:
        return seq

    for token_idx, token_data in enumerate(content_logprobs):
        probs = _extract_logprobs_from_token(token_data)
        if probs:
            seq[token_idx] = probs

    return seq


def process_openai_chat_completion(response: Any, iterations: int) -> list[dict[int, list[float]]]:
    """
    Processes log probabilities from OpenAI Chat Completion (classic 'choices' format).
    Args:
        response: The response object or dictionary from OpenAI API (ChatCompletion).
        iterations: The number of iterations to process.
    Returns:
        list[dict[int, list[float]]]: A list of dictionaries mapping token indices to lists of log probs.
    """
    choices = _get_val(response, "choices", [])
    if not choices:
        return []

    count = min(iterations, len(choices))
    all_sequences = []

    for i in range(count):
        seq_data = _process_single_choice(choices[i])
        all_sequences.append(seq_data)

    return all_sequences


# --- The rest of your code remains unchanged ---


def is_openai_responses_api(outputs: Any) -> bool:
    """Detects the signature of the new OpenAI Responses API."""
    if hasattr(outputs, "object") and outputs.object == "response":
        return True
    if isinstance(outputs, dict) and outputs.get("object") == "response":
        return True
    if hasattr(outputs, "output") or (isinstance(outputs, dict) and "output" in outputs):
        return True
    return False


def process_openai_responses_api(response: Any) -> list[dict[int, list[float]]]:
    """
    Parser for the 'client.responses.create' API.
    Structure: response.output -> [item] -> item.content -> [part] -> part.logprobs
    """
    extracted_batch = []
    output_list = _get_val(response, "output", [])

    for item in output_list:
        seq_logprobs_map: dict[int, list[float]] = {}
        content_list = _get_val(item, "content", [])

        token_index = 0
        for content_part in content_list:
            tokens_data = _get_val(content_part, "logprobs")

            if tokens_data:
                for token_entry in tokens_data:
                    logprobs = _parse_token_entry(token_entry)
                    if logprobs:
                        seq_logprobs_map[token_index] = logprobs
                        token_index += 1

        extracted_batch.append(seq_logprobs_map)

    return extracted_batch


def _parse_token_entry(token_entry: Any) -> list[float]:
    """Parses a single token entry from OpenAI response to extract logprobs."""
    top_k = _get_val(token_entry, "top_logprobs", [])

    if top_k:
        logprobs = []
        for k in top_k:
            k_logprob = _get_val(k, "logprob")
            if k_logprob is not None:
                logprobs.append(float(k_logprob))
        return logprobs
    else:
        m_logprob = _get_val(token_entry, "logprob")
        return [float(m_logprob)] if m_logprob is not None else []
