"""
Generate a dataset with entropy scores for model outputs.

This script generates responses from a language model for a given dataset and
computes entropy-based uncertainty metrics for each generation.

Uses VLLM API to generate outputs and process log probabilities.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from pydantic import BaseModel
from tqdm import tqdm
from vllm import LLM, SamplingParams

from artefactual.calibration.helpers.memory import clear_gpu_memory
from artefactual.calibration.helpers.models import get_model_name, init_llm
from artefactual.preprocessing.vllm_parser import process_vllm_logprobs
from artefactual.scoring.entropy_methods.epr import EPR
from artefactual.utils.io import convert_bytes_to_str, load_tqa_from_json, save_to_json

logger = logging.getLogger(__name__)

# Constants
_SEED = 42
_MAX_NEW_TOKENS = 200
_TOP_P = 1


class GenerationConfig(BaseModel):
    """Configuration for entropy dataset generation."""

    model_path: str = "mistralai/Ministral-8B-Instruct-2410"
    number_logprobs: int = 15
    iterations: int = 1
    top_k_sampling: int = 50
    n_queries: int = -1
    temperature: float = 0.6
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


def _process_results(query_data, outputs, epr_scorer: EPR, *, show_logprobs: bool = False) -> dict:
    """Process outputs and compute EPR scores."""
    query, query_id, expected_answer, answer_aliases = query_data
    list_outputs_text = [output.text for output in outputs[0].outputs]
    expected_answers = [expected_answer, *answer_aliases]

    # Compute EPR score
    epr_scores = epr_scorer.compute(outputs)

    # Add EPR scores to the generated_answers list
    generated_answers_with_scores = []
    for idx, score in enumerate(epr_scores):
        generated_answers_with_scores.append({str(idx): list_outputs_text[idx], "epr_score": score})

    data = {
        "query_id": query_id,
        "query": query,
        "expected_answers": expected_answers,
        "generated_answers": generated_answers_with_scores,
    }

    if show_logprobs:
        # Compute per inner token top_k logprobs
        token_logprob_list: list[dict[int, list[float]]] = process_vllm_logprobs(outputs[0].outputs)
        data["token_logprobs"] = token_logprob_list
    return convert_bytes_to_str(data)


def _prepare_messages(pack: list[tuple[str, str, str, list[str]]]) -> list[list[dict[str, str]]]:
    """
    Prepare messages for generation from a pack of data.

    Args:
        pack (list[tuple[str, str, str, list[str]]]): List of tuples containing (query, query_id, answer, aliases).

    Returns:
        list[list[dict[str, str]]]: List of message lists suitable for chat models.
    """
    all_messages = []
    for query, _, _, _ in pack:
        prompt = (
            "You are a useful assistant that help finding short and precise answers for a given query or question.\n"
            "Please keep your output AS SHORT AND CONCISE AS POSSIBLE.\n"
            f"Here is the query :\n{query}"
        )
        all_messages.append([{"role": "user", "content": prompt}])
    return all_messages


def generate_entropy_dataset(
    input_path: str | Path,
    output_path: str | Path,
    config: GenerationConfig,
) -> Path:
    """
    Generate a dataset with entropy scores for model outputs.

    Args:
        input_path (str | Path): Path to the input dataset.
        output_path (str | Path): Path to save the output dataset.
        config (GenerationConfig): Configuration parameters.

    Returns:
        Path: The path to the generated output file.
    """
    _setup_logging(log_to_file=config.log_to_file)

    torch.cuda.empty_cache()

    model_name = get_model_name(config.model_path)
    logger.info(f"Model name: {model_name}")

    llm: LLM = init_llm(
        model_path=config.model_path,
        seed=_SEED,
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

    pack = load_tqa_from_json(str(input_path))

    # Define output file path once
    input_dataset_name = str(input_path).rsplit("/", maxsplit=1)[-1].rsplit(".")[0]
    output_file = os.path.join(output_path, f"{input_dataset_name}_{model_name}_entropy.json")
    os.makedirs(output_path, exist_ok=True)

    pack_to_process = pack if config.n_queries == -1 else pack[: config.n_queries]

    # Prepare all messages for processing
    all_messages = _prepare_messages(pack_to_process)

    epr_scorer = EPR()

    results = []
    logger.info(f"Generating responses for {len(all_messages)} queries...")
    for i, messages in enumerate(tqdm(all_messages)):
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
        )
        # Process the results for the current query
        query_data = pack_to_process[i]
        result = _process_results(
            query_data,
            outputs,
            epr_scorer,
        )
        results.append(result)
    logger.info("Generation complete.")

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
    return Path(output_file)


if __name__ == "__main__":
    config = GenerationConfig()
    generate_entropy_dataset(
        input_path="/home/gjeannin/artefactual/sample_qa_data.json",
        output_path="outputs/",
        config=config,
    )
