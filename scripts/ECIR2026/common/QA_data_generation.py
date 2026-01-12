import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import ray
import torch
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams

seed = 42

# Parameters
parser = argparse.ArgumentParser(description="Generate answers using an LLM with Ray.")
parser.add_argument(
    "--iterations",
    type=int,
    default=1,
    help="Number of iterations (i.e number of output sequences to return for the given prompt) for sampling.",
)

parser.add_argument(
    "--QA_dataset_path",
    type=str,
    default="trivia_qa.json",
    help="Path to a question answer dataset in JSON format.",
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
    help="Number of queries to process from the dataset. -1 for all.",
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
parser.add_argument(
    "--model_checkpoint",
    type=str,
    required=True,
    help="Path to the Hugging Face model checkpoint."
)
parser.add_argument(
    "--num_gpus_to_use",
    type=int,
    default=4,
    help="Number of GPUs to use for parallel processing."
)
parser.add_argument(
    "--batch_size_per_gpu",
    type=int,
    default=1,
    help="Number of queries processed by each GPU actor simultaneously (vLLM handles internal batching)."
)

args = parser.parse_args()

iterations = args.iterations
top_k_sampling = args.top_k_sampling
n_queries = args.n_queries
number_logprobs = args.number_logprobs
temperature_arg = args.temperature
checkpoint = args.model_checkpoint
num_gpus_to_use = args.num_gpus_to_use
qa_dataset_path = Path(args.QA_dataset_path)

max_new_tokens = 200
top_p = 1
temperatures = [temperature_arg]

# Logging Setup

log_dir = os.path.join("logs")
os.makedirs(log_dir, exist_ok=True)
tz = timezone.utc
log_filename = f'answer_generation_parallel_{datetime.now(tz).strftime("%Y%m%d_%H%M%S")}.log'
log_filepath = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()],
)

if checkpoint is None:
    logging.error("No checkpoint provided. Please specify a model checkpoint.")
    sys.exit(1)

if "/" in checkpoint:
    model_name_suffix = checkpoint.split("/")[-1]
else:
    model_name_suffix = checkpoint.split(".")[-1]
logging.info(f"Using model checkpoint: {checkpoint}")
logging.info(f"Model name for output file: {model_name_suffix}")


# Ray Actor for VLLM

@ray.remote(num_gpus=1)
class VLLMWorker:
    def __init__(self, model_checkpoint: str, seed_val: int, worker_id: int):
        self.worker_id = worker_id
        logging.info(f"Worker {self.worker_id}: Initializing LLM on GPU {ray.get_gpu_ids()}.")

        if "mistralai/" in model_checkpoint:
            self.llm = LLM(
                model=model_checkpoint,
                tokenizer_mode="mistral",
                load_format="mistral",
                config_format="mistral",
                seed=seed_val,
                tensor_parallel_size=1,
            )
        else:
            self.llm = LLM(
                model=model_checkpoint,
                seed=seed_val,
                tensor_parallel_size=1,
            )
        logging.info(f"Worker {self.worker_id}: LLM initialized.")

    def process_query_item(self, query_item: dict, sampling_params: SamplingParams):
        actual_query_text = query_item["question"]
        query_id = query_item["question_id"]
        expected_short_ans = query_item["short_answer"]
        answer_aliases_list = query_item["answer_aliases"]
        # Prompt template :
        PROMPT_MODEL = f"""You are a useful assistant that help finding short and precise answers for a given query or question.
            Please keep your output AS SHORT AND CONCISE AS POSSIBLE.
            Here is the query :
            {actual_query_text}
            """  # Use the variable holding the question text  # noqa: N806
        messages = [{"role": "user", "content": PROMPT_MODEL}]

        logging.info(f"Worker {self.worker_id}: Processing query_id: {query_id} - {actual_query_text[:50]}...")
        timestamp = datetime.now(tz)

        outputs_vllm = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        request_output = outputs_vllm[0]
        list_outputs_text = [output.text for output in request_output.outputs]

        delta_time = (datetime.now(tz) - timestamp).total_seconds()
        logging.info(
            f"Worker {self.worker_id}: Query_id {query_id} generated {len(list_outputs_text)} outputs in {delta_time:.2f}s"
        )

        full_info_list = []
        for k in range(len(request_output.outputs)):
            completion_k = request_output.outputs[k]

            detailed_logprobs_for_completion_k = []
            if completion_k.logprobs:
                for token_idx in range(len(completion_k.logprobs)):
                    logprobs_for_token_dict = completion_k.logprobs[token_idx]

                    single_token_logprob_list = []
                    if logprobs_for_token_dict:
                        for logprob_item in logprobs_for_token_dict.values():
                            single_token_logprob_list.append({
                                "logprob": logprob_item.logprob,
                                "rank": logprob_item.rank,
                                "decoded_token": logprob_item.decoded_token,
                            })
                    detailed_logprobs_for_completion_k.append(single_token_logprob_list)

            full_info_list.append({
                "answer_text": completion_k.text,
                "cumulative_logprob": completion_k.cumulative_logprob,
                "detailed_logprobs": detailed_logprobs_for_completion_k,
            })

        while len(full_info_list) < sampling_params.n and sampling_params.n > 0:
            full_info_list.append({
                "answer_text": "ERROR_MISSING_SEQUENCE",
                "cumulative_logprob": None,
                "detailed_logprobs": [],
            })

        # Construct result_entry using original output field names
        result_entry = {
            "query": actual_query_text,
            "query_id": query_id,
            "expected_answer": expected_short_ans,
            "answer_aliases": answer_aliases_list,
            "generated_answers": [
                {str(i): text} for i, text in enumerate(list_outputs_text)
            ],
            "full_info": full_info_list,
            "processing_time_seconds": delta_time,
            "worker_id": self.worker_id,
            "timestamp": timestamp.isoformat()
        }
        return convert_bytes_to_str(result_entry)

    def shutdown(self):
        logging.info(f"Worker {self.worker_id}: Shutting down.")
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Load the TQA dataset
def load_qa_ds_from_json(input_file=qa_dataset_path):
    try:
        with open(input_file, encoding="utf-8") as f:
            json_data = json.load(f)
        pack_data = [
            {
                "question": item["question"],
                "question_id": item["question_id"],
                "short_answer": item["short_answer"],
                "answer_aliases": item["answer_aliases"],
            }
            for item in json_data
        ]
        logging.info(f"Loaded {len(pack_data)} question-answer pairs from {input_file}")
        return pack_data
    except FileNotFoundError:
        logging.exception(f"File not found: {input_file}")
        return []
    except json.JSONDecodeError:
        logging.exception(f"Error decoding JSON from {input_file}")
        return []


# Bytes to string conversion
def convert_bytes_to_str(obj):
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception as e:
            logging.exception(f"Error decoding bytes: {e}")
            return "ERROR_DECODING_BYTES"
    elif isinstance(obj, dict):
        return {key: convert_bytes_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_bytes_to_str(element) for element in obj]
    return obj


# Main function
if __name__ == "__main__":

    if not torch.cuda.is_available():
        logging.error("CUDA is not available. This script requires GPUs.")
        sys.exit(1)

    available_gpus = torch.cuda.device_count()
    gpus_to_use_actually = min(num_gpus_to_use, available_gpus)
    if gpus_to_use_actually == 0:
        logging.error("No GPUs to use. Exiting.")
        sys.exit(1)
    if gpus_to_use_actually < num_gpus_to_use:
        logging.warning(f"Requested {num_gpus_to_use} GPUs, but only {gpus_to_use_actually} are available/usable.")

    logging.info(f"Initializing Ray to use {gpus_to_use_actually} GPUs.")
    ray.init(num_gpus=gpus_to_use_actually, ignore_reinit_error=True)

    logging.info("Creating VLLM workers...")
    workers = [
        VLLMWorker.remote(model_checkpoint=checkpoint, seed_val=seed, worker_id=i)
        for i in range(gpus_to_use_actually)
    ]
    logging.info(f"Created {len(workers)} VLLM workers.")

    try:
        warmup_sampling_params = SamplingParams(n=1, max_tokens=5, logprobs=number_logprobs)
        logging.info(f"Warming up workers with {number_logprobs=}")

        warmup_tasks = []
        for i, worker in enumerate(workers):
            # KEYS FOR THE WARM-UP CALL DATA
            warmup_query_item = {
                "question": "Test warm-up: What is the capital of France?",
                "question_id": f"warmup_{i}",
                "short_answer": "Paris",
                "answer_aliases": ["Paris"]
            }
            warmup_tasks.append(worker.process_query_item.remote(warmup_query_item, warmup_sampling_params))

        warmup_results = ray.get(warmup_tasks, timeout=300)

        for i, res in enumerate(warmup_results):
            if not res or "ERROR" in str(res):
                logging.warning(f"Worker {i} warm-up may have issues: {str(res)[:200]}")
            else:
                generated_text_key = "0"  # Default key for the first generated answer
                generated_answer_dict = {}
                if res.get("generated_answers") and isinstance(res.get("generated_answers"), list) and len(res.get("generated_answers")) > 0:
                    generated_answer_dict = res.get("generated_answers")[0] if isinstance(res.get("generated_answers")[0], dict) else {}

                logging.info(f"Worker {i} warm-up successful. Generated text: {generated_answer_dict.get(generated_text_key, '')[:50]}...")

        logging.info("All workers responded to warm-up.")

    except ray.exceptions.RayTaskError as e:
        logging.exception(f"A worker task failed critically during warm-up: {e}")
        original_error = e.cause
        logging.exception(f"Original error in actor: {original_error}")
        ray.shutdown()
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Error during worker warm-up: {e}")
        ray.shutdown()
        sys.exit(1)

    for temp_val in temperatures:
        current_temperature = temp_val
        sampling_params = SamplingParams(
            n=iterations,
            max_tokens=max_new_tokens,
            temperature=current_temperature,
            top_p=top_p,
            top_k=top_k_sampling,
            seed=seed,
            logprobs=number_logprobs,
        )

        logging.info("Loading QA pack...")
        pack_dict_list = load_qa_ds_from_json(qa_dataset_path)

        if n_queries > 0:
            pack_to_process = pack_dict_list[:n_queries]
        else:
            pack_to_process = pack_dict_list

        actual_n_queries = len(pack_to_process)

        output_data = {
            "metadata": {
                "generator_model": checkpoint,
                "retriever": "NONE",
                "date": f"{datetime.now(tz)}",
                "temperature": current_temperature,
                "top_k_sampling": top_k_sampling,
                "top_p": top_p,
                "n_queries": actual_n_queries,
                "iterations": iterations,
                "number_logprobs": number_logprobs,
                "num_gpus_used": gpus_to_use_actually,
            },
            "results": [],
        }

        pbar = tqdm(total=actual_n_queries, desc="Processing queries")
        results_futures = []
        worker_idx_cycle = 0

        for i in range(actual_n_queries):
            query_item_data = pack_to_process[i]  # This dict has "question", "short_answer", etc.
            worker = workers[worker_idx_cycle]
            results_futures.append(worker.process_query_item.remote(query_item_data, sampling_params))
            worker_idx_cycle = (worker_idx_cycle + 1) % gpus_to_use_actually

            if len(results_futures) == gpus_to_use_actually or (i == actual_n_queries - 1 and len(results_futures) > 0):
                try:
                    completed_tasks_results = ray.get(results_futures)
                    for result in completed_tasks_results:
                        if result:
                            output_data["results"].append(result)
                    pbar.update(len(completed_tasks_results))
                except ray.exceptions.RayTaskError as e:
                    logging.exception(f"A task in a batch failed: {e}. Query details might be lost for this item. Error: {e.cause}")
                    pbar.update(len(results_futures))
                except Exception as e:
                    logging.exception(f"An unexpected error occurred processing a batch results: {e}")
                    pbar.update(len(results_futures))
                results_futures = []
        pbar.close()

        output_file = (
            f"generated_data/QA_{model_name_suffix}_{current_temperature}_{actual_n_queries}_with_logprobs_parallel.json"
        )
        logging.info(f"Writing results to {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        logging.info(f"Processing for temperature {current_temperature} complete.")

    logging.info("All temperatures processed. Shutting down workers and Ray.")
    if workers:
        ray.get([worker.shutdown.remote() for worker in workers])
    ray.shutdown()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Script finished.")
