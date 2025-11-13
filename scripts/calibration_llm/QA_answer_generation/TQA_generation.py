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

# Add path to import UncertaintyDetector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
from entropyPackage.uncertainty_detector import UncertaintyDetector

EPSILON = 1e-12
seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
# %% Parameters
parser = argparse.ArgumentParser(description="Generate answers using an LLM.")
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
    default=-1,
    help="Number of queries to process from the dataset.",
)
parser.add_argument(
    "--number_logprobs",
    type=int,
    default=15,
    help="Number of log probabilities to request.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for sampling.",
)
parser.add_argument("--model_checkpoint", type=str, help="Path to the Hugging Face model checkpoint.")
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

args = parser.parse_args()

iterations = args.iterations
top_k_sampling = args.top_k_sampling
n_queries = args.n_queries
number_logprobs = args.number_logprobs
temperature = args.temperature
tensor_parallel_size = args.tensor_parallel_size
gpu_memory_utilization = args.gpu_memory_utilization

checkpoint = args.model_checkpoint

max_new_tokens = 200
top_p = 1

temperatures = [1]  # extend the list for temperature scaling.
# %% Load the LLM instance

log_dir = os.path.join("logs")
os.makedirs(log_dir, exist_ok=True)

tz = timezone.utc
log_filename = f"answer_generation_{datetime.now(tz).strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_dir, log_filename)

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
)

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


# %% Load the TQA dataset
def load_tqa_from_json(
    input_file="trivia_qa.json",
):
    """Load the pack data from a JSON file.

    Args:
        input_file (str): Path to the JSON file

    Returns:
        list: List of (question, answer) tuples
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


# %% Bytes to string conversion


def convert_bytes_to_str(obj):
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


# %% Clear memory


def clear_gpu_memory(llm: LLM) -> None:
    # Delete the llm object and free the memory
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully deleted the llm pipeline and freed the GPU memory.")


# %% Main function

if __name__ == "__main__":
    for temperature in temperatures:
        sampling_params = SamplingParams(
            n=iterations,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k_sampling,
            seed=seed,
            logprobs=number_logprobs,
        )

        logging.info("Loading QA pack...")
        pack = load_tqa_from_json("/data/workspace/charles/artefactual_data/web_question_qa.json")

        # Create the complete data structure before writing
        output_data = {
            "metadata": {
                "generator_model": checkpoint,
                "retriever": "NONE",
                "date": f"{datetime.now(tz)}",
                "temperature": temperature,
                "top_k_sampling": top_k_sampling,
                "top_p": top_p,
                "n_queries": n_queries,
                "iterations": iterations,
                "number_logprobs": number_logprobs,
            },
            "results": [],
        }

        for query, query_id, ans, aliases in tqdm(pack[:n_queries]):
            PROMPT_MODEL = f"""You are a useful assistant that help finding short and precise answers for a given query or question.
                Please keep your output AS SHORT AND CONSICE AS POSSIBLE.
                Here is the query :
                {query}
                """
            #  If you don't know the answer, please return : "NONE".
            messages = [{"role": "user", "content": PROMPT_MODEL}]
            logging.info(f"Processing query: {query}")
            timestamp = datetime.now(tz)
            logging.info("Starting model generation...")

            # Generate multiple outputs using vLLM
            outputs = llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                use_tqdm=False,
                # chat_template="/home/cmoslonka/artefactual/qwen3_nonthinking.jinja" if "qwen3" in checkpoint.lower() else None,
                # chat_template_kwargs={"enable_thinking": False},
            )

            # Extract all generated texts
            list_outputs_text = [output.text for output in outputs[0].outputs]

            logging.info(f"Generated {len(list_outputs_text)} outputs")
            delta_time = (datetime.now(tz) - timestamp).total_seconds()
            logging.info(f"Model generation completed in {delta_time:.2f} seconds (for {iterations} runs)")

            # Extract detailed logprobs directly from vLLM outputs
            full_info_list = []

            # Compute EPR scores for all outputs
            epr_scores, token_epr_scores = uncertainty_detector.compute_epr(outputs, return_tokens=True)
            logging.info(f"Computed EPR scores: {epr_scores}")

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
                full_info_list.append({
                    "answer_text": outputs[0].outputs[k].text,
                    "cumulative_logprob": outputs[0].outputs[k].cumulative_logprob,
                    "detailed_logprobs": detailed_logprobs,
                    "epr_score": float(epr_scores[k]),
                    "token_epr": token_epr_scores[k].tolist(),
                })

            result_entry = {
                "query": query,
                "query_id": query_id,
                "expected_answer": ans,
                "answer_aliases": aliases,
                "generated_answers": [{str(i): text} for i, text in enumerate(list_outputs_text)],
                "full_info": full_info_list,
            }
            result_entry = convert_bytes_to_str(result_entry)
            output_data["results"].append(result_entry)

        # Write the complete data structure to file
        output_file = (
            f"/home/gjeannin/artefactual/generation_test/WQA_{name}_{temperature}_{n_queries}_with_logprobs_NNTEST.json"
        )
        logging.info(f"Writing results to {output_file}")

        # Convert any bytes objects to strings before JSON serialization

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

    torch.cuda.empty_cache()
    clear_gpu_memory(llm)
    logging.info("Memory cleared.")
