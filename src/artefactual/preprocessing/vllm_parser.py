from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import CompletionOutput


def process_vllm_logprobs(
    completions: Sequence["CompletionOutput"],
    max_completions: int | None = None,
) -> list[dict[int, list[float]]]:
    """
    Processes log probabilities from vLLM completion outputs.

    Args:
        completions: A sequence of CompletionOutput objects (e.g., from RequestOutput.outputs),
                    each containing log probability data for a generated sequence.
        max_completions: Optional maximum number of completions to process.
                        If None, processes all completions. If specified, processes up to
                        min(max_completions, len(completions)) completions.

    Returns:
        A list of dictionaries, one per processed completion. Each dictionary maps
        token indices (int) to lists of log probabilities (list[float]) for the top-k
        tokens at that position in the sequence.

    Examples:
        >>> # Process all completions from a RequestOutput
        >>> result = llm.chat(messages, sampling_params)
        >>> logprobs = process_vllm_logprobs(result[0].outputs)
        >>>
        >>> # Process only the first 3 completions
        >>> logprobs = process_vllm_logprobs(result[0].outputs, max_completions=3)
    """
    if not completions:
        return []

    all_sequences = []

    # Determine how many completions to process
    num_to_process = len(completions)
    if max_completions is not None:
        num_to_process = min(max_completions, num_to_process)

    for i in range(num_to_process):
        seq = {}
        token_logprobs = completions[i].logprobs

        if not token_logprobs:
            all_sequences.append(seq)
            continue

        for token_idx, token_topk_log_dict in enumerate(token_logprobs):
            topk_logprobs_list = [inner_token.logprob for inner_token in token_topk_log_dict.values()]

            seq[token_idx] = topk_logprobs_list

        all_sequences.append(seq)

    return all_sequences
