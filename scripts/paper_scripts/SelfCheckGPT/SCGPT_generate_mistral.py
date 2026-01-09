# generate.py
import argparse
import json
import logging
import os
from datetime import datetime, timezone

from tqdm import tqdm
from vllm import LLM, SamplingParams

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tz = timezone.utc


# --- Helper Functions ---
def load_tqa_from_json(input_file="trivia_qa.json"):
    """Load the dataset from a JSON file."""
    try:
        with open(input_file, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {input_file}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {input_file}")
        return []


def main(args):
    """
    Main function to generate text using a vLLM model and save the results.
    """
    # --- Initialize LLM ---
    logging.info(f"Loading Mistral model: {args.model_checkpoint}")
    llm = LLM(
                model=args.model_checkpoint,
                tokenizer_mode="mistral",  # For mistral models
                load_format="mistral",
                config_format="mistral",
                seed=args.seed,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=8196,
            )

    # --- Set Sampling Parameters ---
    # The first output is the primary response, the rest are samples.
    sampling_params = SamplingParams(
        n=args.iterations,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    # --- Load Data ---
    logging.info(f"Loading data from {args.input_file}")
    pack = load_tqa_from_json(args.input_file)
    if not pack:
        return

    # Slice the dataset if n_queries is specified
    queries_to_process = pack[: args.n_queries]

    # --- Prepare Output Structure ---
    output_data = {
        "metadata": {
            "generator_model": args.model_checkpoint,
            "date": f"{datetime.now(tz)}",
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "n_queries": len(queries_to_process),
            "iterations_per_query": args.iterations,
        },
        "results": [],
    }

    # --- Generation Loop ---
    logging.info(f"Starting generation for {len(queries_to_process)} queries...")
    for item in tqdm(queries_to_process, desc="Generating Responses"):
        query = item["question"]
        prompt = (
            "You are a useful assistant that help finding short and precise answers for a given query or question.\n"
            "Please keep your output AS SHORT AND CONSICE AS POSSIBLE.\n"
            f"Here is the query:\n{query}"
        )
        messages = [{"role": "user", "content": prompt}]

        # Generate multiple outputs using vLLM
        outputs = llm.chat(
            messages=messages,  # pyright: ignore[reportArgumentType]
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # The first generation is the main answer, the rest are samples
        main_answer = outputs[0].outputs[0].text
        sampled_answers = [output.text for output in outputs[0].outputs[1:]]

        result_entry = {
            "query": query,
            "query_id": item.get("question_id"),
            "expected_answer": item.get("short_answer"),
            "answer_aliases": item.get("answer_aliases"),
            "main_answer": main_answer,
            "sampled_answers": sampled_answers,
        }
        output_data["results"].append(result_entry)

    # --- Save Results ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Writing results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logging.info("Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLM responses for a given dataset.")
    parser.add_argument(
        "--model_checkpoint", type=str, default="mistralai/Ministral-8B-Instruct-2410", help="Hugging Face model checkpoint."
    )
    parser.add_argument("--input_file", type=str, default="trivia_qa.json", help="Path to the input JSON file.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="generated_data/SCGPT_Mistral_8B_10000.json",
        help="Path to save the output JSON file.",
    )
    parser.add_argument("--n_queries", type=int, default=10000, help="Number of queries to process from the input file.")
    parser.add_argument("--iterations", type=int, default=11, help="Total generations per query (1 main + 10 samples).")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1, help="Top-p (nucleus) sampling.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")

    args = parser.parse_args()
    main(args)
