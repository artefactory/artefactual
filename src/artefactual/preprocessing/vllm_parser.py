from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import RequestOutput


def process_vllm_logprobs(outputs: list["RequestOutput"], iterations: int) -> list[dict[int, list[float]]]:
    """
    Processes log probabilities from vllm.chat outputs for a given number of iterations.
    Args:
        outputs (list[RequestOutput]): A list containing model output objects, each with log probability data.
        iterations (int): The number of iterations to process, corresponding to the number of output sequences.
    Returns:
        list[dict[int, list[float]]]: A list of dictionaries mapping token indices to lists of log probs
                                      for each token in the sequence.
    """

    if not outputs or not outputs[0].outputs:
        return []

    all_sequences = []

    for i in range(iterations):
        seq = {}
        token_logprobs = outputs[0].outputs[i].logprobs

        if not token_logprobs:
            all_sequences.append(seq)
            continue

        for token_idx, token_topk_log_dict in enumerate(token_logprobs):
            topk_logprobs_list = [inner_token.logprob for inner_token in token_topk_log_dict.values()]

            seq[token_idx] = topk_logprobs_list

        all_sequences.append(seq)

    return all_sequences
