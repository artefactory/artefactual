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
            # token_idx is the position in the sequence
            # token_topk_log_dict is a dict of top-K logprobs for that token,
            # where the first item is the sampled token. (rank can be > k)
            topk_logprobs_with_rank = [
                (inner_token.logprob, inner_token.rank) for inner_token in token_topk_log_dict.values()
            ]
            # Sort by rank (ascending)
            sorted_topk = sorted(topk_logprobs_with_rank, key=lambda x: x[1], reverse=False)
            seq[token_idx] = [logp for (logp, rank) in sorted_topk]

        all_sequences.append(seq)

    return all_sequences
