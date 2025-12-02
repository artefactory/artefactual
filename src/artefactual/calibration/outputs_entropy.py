"""
Generate a dataset with entropy scores for model outputs.

This script generates responses from a language model for a given dataset and
computes entropy-based uncertainty metrics for each generation.
"""

import logging
import os
from datetime import datetime, timezone

import torch
from pydantic import BaseModel
from tqdm import tqdm
from vllm import LLM, SamplingParams

from artefactual.data.data_model import Completion
from artefactual.scoring.epr import EPR
from artefactual.utils.io import convert_bytes_to_str, load_tqa_from_json, save_to_json
from artefactual.utils.memory import clear_gpu_memory
from artefactual.utils.models import get_model_name, init_llm

logger = logging.getLogger(__name__)

# Constants
_SEED = 42
_MAX_NEW_TOKENS = 200
_TOP_P = 1


class GenerationConfig(BaseModel):
    """Configuration for entropy dataset generation."""

    model_path: str = "mistralai/Ministral-8B-Instruct-2410"
    number_logprobs: int = 15
    iterations: int = 2
    top_k_sampling: int = 50
    n_queries: int = -1
    temperature: float = 0.6
    tensor_parallel_size: int = 1
    log_to_file: bool = False


def _setup_logging(*, log_to_file: bool) -> None:
    """Configure logging to file if requested."""
    if log_to_file:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"generation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger("").addHandler(file_handler)
        logging.getLogger("").setLevel(logging.INFO)


def _process_logprobs(outputs, iterations):
    """Extract top-K logprobs per token, as a dict indexed by token position."""

    all_sequences = []

    for k in range(iterations):
        seq_dict = {}

        # vLLM format: list[dict{rank→RankLogProb}]
        token_logprobs = outputs[0].outputs[k].logprobs
        if not token_logprobs:
            all_sequences.append(seq_dict)
            continue

        for token_idx, topk_dict in enumerate(token_logprobs):
            # Extract top-K logprobs for this token
            topk_logprobs = [rank_obj.logprob for rank_obj in topk_dict.values()]

            seq_dict[token_idx] = topk_logprobs

        all_sequences.append(seq_dict)

    return all_sequences


def _prepare_messages(pack):
    """Prepare messages for batch generation."""
    all_messages = []
    for query, _, _, _ in pack:
        prompt = (
            "You are a useful assistant that help finding short and precise answers for a given query or question.\n"
            "Please keep your output AS SHORT AND CONSICE AS POSSIBLE.\n"
            f"Here is the query :\n{query}"
        )
        all_messages.append([{"role": "user", "content": prompt}])
    return all_messages


def _process_batch_results(pack, batch_outputs, iterations, epr_scorer):
    """Process batch outputs and compute EPR scores."""
    results = []
    for i, (query, query_id, expected_answer, answer_aliases) in enumerate(tqdm(pack)):
        outputs = [batch_outputs[i]]
        list_outputs_text = [output.text for output in outputs[0].outputs]
        expected_answers = [expected_answer, *answer_aliases]

        token_logprob_list = _process_logprobs(outputs, iterations)

        # Parse completions and compute EPR scores
        completions_for_query = [
            Completion.model_validate({"token_logprobs": seq_dict}) for seq_dict in token_logprob_list
        ]
        epr_scores = epr_scorer.compute(completions_for_query)

        # Add EPR scores to the generated_answers list
        generated_answers_with_scores = []
        for idx, score in enumerate(epr_scores):
            generated_answers_with_scores.append({str(idx): list_outputs_text[idx], "epr_score": score})

        data = {
            "query_id": query_id,
            "query": query,
            "expected_answers": expected_answers,
            "generated_answers": generated_answers_with_scores,
            "token_logprobs": token_logprob_list,
        }
        results.append(convert_bytes_to_str(data))
    return results


def generate_entropy_dataset(
    input_path: str,
    output_path: str,
    config: GenerationConfig,
) -> None:
    """
    Generate a dataset with entropy scores for model outputs.

    Args:
        input_path (str): Path to the input dataset.
        output_path (str): Path to save the output dataset.
        config (GenerationConfig): Configuration parameters.
    """
    _setup_logging(log_to_file=config.log_to_file)

    torch.cuda.empty_cache()

    model_name = get_model_name(config.model_path)
    logger.info(f"Model name: {model_name}")

    llm: LLM = init_llm(
        model_path=config.model_path,
        seed=_SEED,
        tensor_parallel_size=config.tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        n=config.iterations,
        max_tokens=_MAX_NEW_TOKENS,
        temperature=config.temperature,
        top_p=_TOP_P,
        top_k=config.top_k_sampling,
        seed=_SEED,
        logprobs=config.number_logprobs,
    )

    pack = load_tqa_from_json(input_path)

    # Define output file path once
    input_dataset_name = input_path.rsplit("/", maxsplit=1)[-1].rsplit(".")[0]
    output_file = os.path.join(output_path, f"{input_dataset_name}_{model_name}_entropy.json")
    os.makedirs(output_path, exist_ok=True)

    pack_to_process = pack if config.n_queries == -1 else pack[: config.n_queries]

    # Instantiate the EPR scorer
    epr_scorer = EPR(k=config.number_logprobs)

    # Prepare all messages for batch processing
    all_messages = _prepare_messages(pack_to_process)

    # Generate all outputs in a single batch call
    logger.info(f"Generating responses for {len(all_messages)} queries in a batch...")
    batch_outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        use_tqdm=True,  # Use vLLM's internal tqdm for batch progress
    )
    logger.info("Batch generation complete.")

    # Process the batch outputs
    results = _process_batch_results(pack_to_process, batch_outputs, config.iterations, epr_scorer)

    # Create the complete data structure
    raw_data = {
        "metadata": {
            "generator_model": config.model_path,
            "retriever": "NONE",
            "date": f"{datetime.now(timezone.utc)}",
            "temperature": config.temperature,
            "top_k_sampling": config.top_k_sampling,
            "top_p": _TOP_P,
            "n_queries": config.n_queries,
            "iterations": config.iterations,
            "number_logprobs": config.number_logprobs,
        },
        "results": results,
    }

    # Save the entire dataset to the JSON file once after the loop
    logger.info(f"Saving results to {output_file}")
    save_to_json(raw_data, output_file)

    clear_gpu_memory(llm)


if __name__ == "__main__":
    config = GenerationConfig(tensor_parallel_size=1)
    generate_entropy_dataset(
        input_path="/home/gjeannin/artefactual/sample_qa_data.json",
        output_path="outputs/",
        config=config,
    )
