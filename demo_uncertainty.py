#!/usr/bin/env python3
"""Demo script for the UncertaintyDetector class.

This script demonstrates how to use the UncertaintyDetector to compute
Entropy Production Rate (EPR) scores for language model outputs with vLLM.

Usage:
    python demo_uncertainty.py --model_checkpoint MODEL [--n_queries N] [--iterations I] [--number_logprobs K]

Examples:
    # Run with Mistral model on 5 queries
    python demo_uncertainty.py --model_checkpoint mistralai/Mistral-7B-Instruct-v0.2 --n_queries 5
    
    # Run with custom parameters
    python demo_uncertainty.py --model_checkpoint MODEL --n_queries 10 --iterations 5 --number_logprobs 15
"""

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone

import ray
import torch
from tqdm import tqdm
from vllm import LLM
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.sampling_params import SamplingParams

import numpy as np

from artefactual.models.uncertainty import UncertaintyDetector

EPSILON = 1e-12
seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"

# %% Parameters
parser = argparse.ArgumentParser(description="Demo: Generate answers and compute EPR scores using UncertaintyDetector.")
parser.add_argument(
    "--iterations",
    type=int,
    default=10,
    help="Number of iterations for sampling.",
)
parser.add_argument(
    "--top_k_sampling",
    type=int,
    default=50,
    help="Top K for sampling during generation.",
)
parser.add_argument(
    "--n_queries",
    type=int,
    default=5,
    help="Number of queries to process from the dataset.",
)
parser.add_argument(
    "--number_logprobs",
    type=int,
    default=15,
    help="Number of log probabilities to request (K for UncertaintyDetector).",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for sampling.",
)
parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the Hugging Face model checkpoint.")
parser.add_argument(
    "--tensor_parallel_size",
    type=int,
    default=2,
    help="Tensor parallel size for model parallelism.",
)
parser.add_argument(
    "--gpu_memory_utilization",
    type=float,
    default=0.90,
    help="GPU memory utilization (0.0 to 1.0).",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="sample-qa-data.json",
    help="Path to the question-answer JSON file.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="generation_test",
    help="Directory to save output JSON files.",
)

args = parser.parse_args()

iterations = args.iterations
top_k_sampling = args.top_k_sampling
n_queries = args.n_queries
number_logprobs = args.number_logprobs
temperature = args.temperature
tensor_parallel_size = args.tensor_parallel_size
gpu_memory_utilization = args.gpu_memory_utilization
checkpoint = args.model_checkpoint
data_path = args.data_path
output_dir = args.output_dir

max_new_tokens = 200
top_p = 1

temperatures = [temperature]  # Can extend for temperature scaling


# %% Setup logging

log_dir = os.path.join("logs")
os.makedirs(log_dir, exist_ok=True)

tz = timezone.utc
log_filename = f"demo_uncertainty_{datetime.now(tz).strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_dir, log_filename)

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
)


# %% Initialize model

torch.cuda.empty_cache()

if checkpoint is None:
    logging.error("No checkpoint provided. Please specify a model checkpoint.")
    sys.exit(1)

# Get model name from checkpoint
if "/" in checkpoint:
    name = checkpoint.split("/")[-1]
else:
    name = checkpoint.split(".")[-1]
    
logging.info(f"Using model checkpoint: {checkpoint}")
logging.info(f"Model name: {name}")

# Specific config for Mistral models
if checkpoint in {"mistralai/Ministral-8B-Instruct-2410", "mistralai/Mistral-Small-3.1-24B-Instruct-2503"}:
    llm = LLM(
        model=checkpoint,
        tokenizer_mode="mistral",  # For mistral models
        load_format="mistral",
        config_format="mistral",
        seed=seed,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
else:
    llm = LLM(
        model=checkpoint,
        seed=seed,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

# Initialize UncertaintyDetector with K matching number_logprobs
uncertainty_detector = UncertaintyDetector(K=number_logprobs)
logging.info(f"Initialized UncertaintyDetector with K={number_logprobs}")


# %% Load dataset


def load_tqa_from_json(input_file: str):
    """Load the QA data from a JSON file.

    Args:
        input_file (str): Path to the JSON file

    Returns:
        list: List of (question, question_id, short_answer, answer_aliases) tuples
    """
    try:
        with open(input_file, encoding="utf-8") as f:
            json_data = json.load(f)

        # Convert dictionaries back to tuples
        pack_data = [
            (
                item["question"],
                item["question_id"],
                item["short_answer"],
                item["answer_aliases"],
            )
            for item in json_data
        ]
        logging.info(f"Loaded {len(pack_data)} question-answer pairs from {input_file}")
        return pack_data

    except FileNotFoundError:
        logging.error(f"File not found: {input_file}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {input_file}")
        return []


# %% Utility functions


def convert_bytes_to_str(obj):
    """Convert bytes objects to strings recursively in nested structures."""
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception as e:
            logging.error(f"Error decoding bytes: {e}")
            return "ERROR"
    elif isinstance(obj, dict):
        return {key: convert_bytes_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_bytes_to_str(element) for element in obj]
    return obj


def clear_gpu_memory(llm: LLM) -> None:
    """Clear GPU memory after model usage."""
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    logging.info("Successfully deleted the llm pipeline and freed the GPU memory.")


# %% Main processing loop

if __name__ == "__main__":
    for temp in temperatures:
        sampling_params = SamplingParams(
            n=iterations,
            max_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            top_k=top_k_sampling,
            seed=seed,
            logprobs=number_logprobs,
        )

        logging.info("=" * 80)
        logging.info("DEMO: Uncertainty Detection with UncertaintyDetector")
        logging.info("=" * 80)
        logging.info(f"Parameters: iterations={iterations}, K={number_logprobs}, temperature={temp}")
        logging.info("")

        logging.info("Loading QA dataset...")
        pack = load_tqa_from_json(data_path)

        if not pack:
            logging.error("Failed to load data. Exiting.")
            sys.exit(1)

        # Create the complete data structure before writing
        output_data = {
            "metadata": {
                "generator_model": checkpoint,
                "retriever": "NONE",
                "date": f"{datetime.now(tz)}",
                "temperature": temp,
                "top_k_sampling": top_k_sampling,
                "top_p": top_p,
                "n_queries": n_queries,
                "iterations": iterations,
                "number_logprobs": number_logprobs,
                "uncertainty_detector_K": number_logprobs,
            },
            "results": [],
        }

        logging.info(f"Processing {min(n_queries, len(pack))} queries...")
        logging.info("")

        for query, query_id, ans, aliases in tqdm(pack[:n_queries]):
            PROMPT_MODEL = f"""You are a useful assistant that helps finding short and precise answers for a given query or question.
Please keep your output AS SHORT AND CONCISE AS POSSIBLE.
Here is the query:
{query}
"""
            messages = [{"role": "user", "content": PROMPT_MODEL}]
            logging.info(f"Processing query: {query}")
            timestamp = datetime.now(tz)
            logging.info("Starting model generation...")

            # Generate multiple outputs using vLLM
            outputs = llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            # Extract all generated texts
            list_outputs_text = [output.text for output in outputs[0].outputs]

            logging.info(f"Generated {len(list_outputs_text)} outputs")
            delta_time = (datetime.now(tz) - timestamp).total_seconds()
            logging.info(f"Model generation completed in {delta_time:.2f} seconds (for {iterations} runs)")

            # === UNCERTAINTY DETECTION ===
            logging.info("Computing EPR scores with UncertaintyDetector...")
            timestamp_epr = datetime.now(tz)
            
            mock_outputs = [
                type('obj', (object,), {'outputs': [outputs[0].outputs[k]]})()
                for k in range(iterations)
            ]

            # Compute all EPR scores in a single, more efficient call
            epr_scores_list, all_token_scores = uncertainty_detector.compute_epr(mock_outputs, return_tokens=True)

            # Extract detailed logprobs directly from vLLM outputs and compute EPR for each
            full_info_list = []            
            for k in range(iterations):
                # Extract detailed logprobs from vLLM output
                token_logprobs_raw = outputs[0].outputs[k].logprobs
                detailed_logprobs = [
                    [
                        {
                            "logprob": list(token_logprobs_raw[i].values())[rank].logprob,
                            "rank": list(token_logprobs_raw[i].values())[rank].rank,
                            "decoded_token": list(token_logprobs_raw[i].values())[rank].decoded_token,
                        }
                        for rank in range(len(token_logprobs_raw[i]))
                    ]
                    for i in range(len(token_logprobs_raw))
                ]

                # Add to full_info with EPR scores
                full_info_list.append(
                    {
                        "answer_text": outputs[0].outputs[k].text,
                        "cumulative_logprob": outputs[0].outputs[k].cumulative_logprob,
                        "detailed_logprobs": detailed_logprobs,
                        "epr_score": float(epr_scores_list[k]),
                        "token_epr": all_token_scores[k].tolist(),
                    }

                )
            
            delta_epr = (datetime.now(tz) - timestamp_epr).total_seconds()
            logging.info(f"EPR computation completed in {delta_epr:.4f} seconds")
            logging.info(f"EPR scores: {epr_scores_list}")
            
            # Compute EPR statistics
            epr_stats = {
                "mean": float(np.mean(epr_scores_list)),
                "std": float(np.std(epr_scores_list)),
                "min": float(np.min(epr_scores_list)),
                "max": float(np.max(epr_scores_list)),
                "median": float(np.median(epr_scores_list)),
            }
            logging.info(f"EPR statistics: mean={epr_stats['mean']:.4f}, std={epr_stats['std']:.4f}, "
                        f"min={epr_stats['min']:.4f}, max={epr_stats['max']:.4f}")
            logging.info("")

            result_entry = {
                "query": query,
                "query_id": query_id,
                "expected_answer": ans,
                "answer_aliases": aliases,
                "generated_answers": [{str(i): text} for i, text in enumerate(list_outputs_text)],
                "full_info": full_info_list,
                "epr_statistics": epr_stats,
            }
            result_entry = convert_bytes_to_str(result_entry)
            output_data["results"].append(result_entry)

        # Write the complete data structure to file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"DEMO_{name}_{temp}_{n_queries}_with_EPR.json")
        logging.info(f"Writing results to {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        logging.info("✓ Results saved successfully!")

    torch.cuda.empty_cache()
    clear_gpu_memory(llm)
    logging.info("Memory cleared.")
    logging.info("")
    logging.info("=" * 80)
    logging.info("DEMO COMPLETED SUCCESSFULLY!")
    logging.info("=" * 80)

