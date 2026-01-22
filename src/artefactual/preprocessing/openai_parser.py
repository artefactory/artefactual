from typing import Any


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely retrieves a value from an object attribute or a dictionary key.

    Args:
        obj (Any): The object or dictionary to retrieve the value from.
        key (str): The attribute name or dictionary key to look up.
        default (Any, optional): The value to return if the key or attribute is not found. Defaults to None.

    Returns:
        Any: The retrieved value or the default value.
    """
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _extract_logprobs_from_token(token_data: Any) -> list[float]:
    """
    Extracts log probabilities from a single token's data structure.

    Args:
        token_data (Any): The data structure containing token information, expected to have 'top_logprobs'.

    Returns:
        list[float]: A list of log probability values extracted from the token data.
    """
    top_logprobs = _get_val(token_data, "top_logprobs", [])

    if not top_logprobs:
        return []

    probs = []
    for item in top_logprobs:
        val = _get_val(item, "logprob")
        if val is not None:
            probs.append(float(val))
    return sorted(probs, reverse=True)  # Sort in descending order : highest logprob first


def _process_single_choice(choice: Any) -> dict[int, list[float]]:
    """
    Processes log probabilities for a single choice object from the API response.

    Args:
        choice (Any): The choice object containing log probability information.

    Returns:
        dict[int, list[float]]: A dictionary mapping token indices to lists of log probabilities.
    """
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
        response (Any): The response object or dictionary from OpenAI API (ChatCompletion).
        iterations (int): The number of iterations (choices) to process.

    Returns:
        list[dict[int, list[float]]]: A list of dictionaries, where each dictionary maps token indices to lists of
        log probabilities for a sequence.
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


def is_openai_responses_api(outputs: Any) -> bool:
    """
    Detects if the output follows the signature of the new OpenAI Responses API.

    Args:
        outputs (Any): The output object or dictionary to inspect.

    Returns:
        bool: True if the output matches the OpenAI Responses API signature, False otherwise.
    """
    if hasattr(outputs, "object") and outputs.object == "response":
        return True
    if isinstance(outputs, dict) and outputs.get("object") == "response":
        return True
    return hasattr(outputs, "output") or (isinstance(outputs, dict) and "output" in outputs)


def process_openai_responses_api(response: Any) -> list[dict[int, list[float]]]:
    """
    Parses the response from the 'client.responses.create' API to extract log probabilities.

    Structure expected: response.output -> [item] -> item.content -> [part] -> part.logprobs

    Args:
        response (Any): The response object from the OpenAI Responses API.

    Returns:
        list[dict[int, list[float]]]: A list of dictionaries, where each dictionary maps token indices to lists of log
        probabilities for a sequence.
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
    """
    Parses a single token entry from an OpenAI response to extract log probabilities.

    Args:
        token_entry (Any): The token entry object containing log probability data.

    Returns:
        list[float]: A list of log probabilities associated with the token.
    """
    top_k = _get_val(token_entry, "top_logprobs", [])

    if top_k:
        logprobs = []
        for k in top_k:
            k_logprob = _get_val(k, "logprob")
            if k_logprob is not None:
                logprobs.append(float(k_logprob))
        return sorted(logprobs, reverse=True)  # Sort in descending order : highest logprob first
    m_logprob = _get_val(token_entry, "logprob")
    return [float(m_logprob)] if m_logprob is not None else []
