import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Vocabulary size for different models

MINISTRAL_8B_VOCAB_SIZE = 2**17
FALCON_10B_VOCAB_SIZE = 2**17
QWEN_25_3B_VOCAB_SIZE = 151936

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

parser = argparse.ArgumentParser(description="Process a JSON file provided via command line.")

parser.add_argument(
    "--input_file",
    "-i",
    dest="input_filename",
    required=True,
    type=str,
    help="Path to the input JSON file to be processed.",
)
parser.add_argument(
    "--output_file",
    "-o",
    dest="output_filename",
    required=False,
    type=str,
    help="Path to the output file where results will be saved.",
)

args = parser.parse_args()

input_file = Path(args.input_filename)
output_file_name = args.output_filename  # Access using the 'dest' name or the long flag name without '--'
if not output_file_name:
    output_file_name = input_file.stem + "_entropy_score.json"  # Default output file name
    output_file_name = Path(input_file.parent, output_file_name)  # Use the same directory as input file


def compute_entropy_rate(logprobs, number_logprobs, vocab_size: int):
    entropy_rate = 0
    token_relative_remaining_entropy = []
    for token in logprobs:  # length of the answer
        token_logprobs = np.array([rank["logprob"] for rank in token])  # logprobs of the tokens
        token_probs = np.exp(token_logprobs)
        sum_probs = np.sum(token_probs)  # compute the sum of probabilities over the top k tokens
        remaining_prob = 1 - sum_probs  # remaining probability to be distributed over the rest of the vocabulary

        token_entropy = -np.sum(token_probs * np.log2(token_probs + 1e-20))  # Avoid log(0) by adding a small value
        entropy_rate += token_entropy

        max_remaining_entropy = (
            -remaining_prob * np.log2(remaining_prob) + np.log2(vocab_size - number_logprobs) * remaining_prob
        )
        token_relative_remaining_entropy.append(max_remaining_entropy / token_entropy)
    return entropy_rate, entropy_rate / len(logprobs), token_relative_remaining_entropy


logging.info(f"Processing file: {input_file}")

if __name__ == "__main__":
    json_data = None  # init
    try:
        with open(input_file, encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)  # Exit if file doesn't exist
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {input_file}: {e}")
        sys.exit(1)  # Exit if JSON is invalid
    except Exception as e:
        logging.error(f"An unexpected error occurred opening or reading {input_file}: {e}")
        sys.exit(1)  # Exit on other read errors

    if json_data:  # Proceed only if file read was successful
        try:
            metadata = json_data["metadata"]
            json_results = json_data["results"]
            # json_results_full_info = [item["full_info"] for item in json_results]

        except KeyError as e:
            # Log the specific key error and then exit
            logging.error(f"Check JSON file structure. Required key missing: {e}")
            sys.exit(1)  # <--- Add this line to stop the program
        except TypeError as e:
            # Handle cases like json_data["results"] not being iterable or items not being dicts
            logging.error(f"Check JSON file structure. Type error encountered: {e}")
            sys.exit(1)  # <--- Also exit on unexpected types
        except Exception as e:
            # Catch any other unexpected errors during processing
            logging.error(f"An unexpected error occurred processing JSON data: {e}")
            sys.exit(1)  # <--- Exit on other processing errors

    # get vocab_size from metadata
    generator_model = metadata["generator_model"]
    number_logprobs = (
        metadata["number_logprobs"] if "number_logprobs" in metadata else len(json_results[0]["full_info"][0])
    )
    number_iterations = (
        metadata["iterations"] if "iterations" in metadata else len(json_results[0]["generated_answers"])
    )

    if generator_model == "mistralai/Ministral-8B-Instruct-2410":
        vocab_size = MINISTRAL_8B_VOCAB_SIZE
    elif generator_model == "tiiuae/Falcon3-10B-Instruct":
        vocab_size = FALCON_10B_VOCAB_SIZE
    elif generator_model == "Qwen/Qwen2.5-3B-Instruct":
        vocab_size = QWEN_25_3B_VOCAB_SIZE
    else:
        logging.error(f"Unknown generator model: {generator_model}. Please specify a known model in JSON metadata.")
        print("-" * 30)
        vocab_size = input("Please specify the vocabulary size of the model used (this needs to be an integer): ")
        try:
            vocab_size = int(vocab_size)
        except ValueError:
            logging.error(f"Invalid vocabulary size provided: {vocab_size}. Please try again.")
        vocab_size = input("Please specify the vocabulary size of the model used (this needs to be an integer): ")
        try:
            vocab_size = int(vocab_size)
        except ValueError:
            logging.error(f"Invalid vocabulary size provided: {vocab_size}. Exiting.")
            sys.exit(1)
        if vocab_size <= 0:
            logging.error(f"Vocabulary size must be a positive integer. Provided: {vocab_size}. Exiting.")
            sys.exit(1)

    results = []

    for item in json_results:
        query = item["query"]
        expected_answer = item["expected_answer"]
        full_info = item["full_info"]
        scored_answers = []
        for ans in full_info:
            answer_text = ans["answer_text"]
            cumulative_logprob = ans["cumulative_logprob"]
            logprobs = ans["detailed_logprobs"]
            entropy_rate, entropy_rate_per_token, token_relative_remaining_entropy = compute_entropy_rate(
                logprobs, number_logprobs, vocab_size
            )
            scored_answer = {
                "answer_text": answer_text,
                "cumulative_logprob": cumulative_logprob,
                "entropy_rate": entropy_rate,
                "entropy_rate_per_token": entropy_rate_per_token,
                "token_relative_remaining_entropy": token_relative_remaining_entropy,
            }
            scored_answers.append(scored_answer)
        results.append({"query": query, "expected_answer": expected_answer, "scored_answers": scored_answers})
    # Save results to output file
    try:
        with open(output_file_name, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to {output_file_name}")
    except Exception as e:
        logging.error(f"An error occurred while saving results to {output_file_name}: {e}")
        sys.exit(1)
