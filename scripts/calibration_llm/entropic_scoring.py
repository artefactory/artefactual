import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Vocabulary size for different models

MINISTRAL_8B_VOCAB_SIZE = 2**17
FALCON_10B_VOCAB_SIZE = 2**17
QWEN_25_3B_VOCAB_SIZE = 151936

EPSILON = 1e-20  # Small value to avoid log(0) in entropy calculations

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

parser = argparse.ArgumentParser(description="Process a JSON file provided via command line.")

parser.add_argument(
    "--input_file",
    "-i",
    dest="input_filename",
    required=True,
    type=Path,
    help="Path to the input JSON file to be processed.",
)
parser.add_argument(
    "--output_file",
    "-o",
    dest="output_filename",
    required=False,
    type=Path,
    help="Path to the output file where results will be saved.",
)

args = parser.parse_args()

input_file = args.input_filename
output_file_name = args.output_filename  # Access using the 'dest' name or the long flag name without '--'
if not output_file_name:
    output_file_name = input_file.stem + "_entropy_score.json"  # Default output file name
    output_file_name = Path(input_file.parent, output_file_name)  # Use the same directory as input file


def compute_entropy_rate(
    logprobs: list[list[dict[str, Any]]], number_logprobs: int, vocab_size: int
) -> tuple[float, float, list[float]]:
    """
    Computes entropy metrics for a sequence of token log probabilities.

    Args:
        logprobs: A list where each element corresponds to a token in the sequence.
                  Each element is itself a list of dictionaries, where each dictionary
                  represents one of the top-k logprobs provided for that token position
                  (e.g., [{'token_str': ' a', 'logprob': -1.2}, ...]).
                  Expected keys in the inner dict: 'logprob'.
        number_logprobs: The number of logprobs provided per token (k in top-k).
        vocab_size: The total vocabulary size of the language model.

    Returns:
        A tuple containing:
        - cumulative_entropy_rate (float): The sum of token entropies over the sequence.
        - average_entropy_rate_per_token (float): The cumulative entropy rate divided by the number of tokens.
        - token_relative_remaining_entropy (List[float]): A list containing the ratio of
          estimated maximum remaining entropy to calculated token entropy for each token position.
    """
    cumulative_entropy_rate = 0.0
    token_relative_remaining_entropy = []
    if not logprobs:
        return 0.0, 0.0, []
    for token in logprobs:  # length of the answer
        token_logprobs = np.array([rank["logprob"] for rank in token])  # logprobs of the tokens
        token_probs = np.exp(token_logprobs)
        sum_probs = np.sum(token_probs)  # compute the sum of probabilities over the top k tokens

        # Clamp sum_probs to avoid remaining_prob being negative due to float precision
        sum_probs = min(sum_probs, 1.0)
        remaining_prob = 1.0 - sum_probs

        token_entropy = -np.sum(token_probs * np.log2(token_probs + EPSILON))  # Avoid log(0) by adding a small value
        cumulative_entropy_rate += token_entropy

        max_remaining_entropy = (
            -remaining_prob * np.log2(remaining_prob + EPSILON) + np.log2(vocab_size - number_logprobs) * remaining_prob
        )  # Supposing uniform distribution over the remaining tokens (very bold assumption)
        if vocab_size - number_logprobs <= 0:
            logging.warning("Vocab size is too small compared to the number of logprobs. The remaining entropy calculation might be inaccurate.")
        if token_entropy <= EPSILON:
            relative_remaining = np.inf
        else:
            relative_remaining = max_remaining_entropy / token_entropy
        token_relative_remaining_entropy.append(relative_remaining)

    return cumulative_entropy_rate, cumulative_entropy_rate / len(logprobs), token_relative_remaining_entropy


if __name__ == "__main__":
    logging.info(f"Processing file: {input_file}")
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

    if not json_results:
        logging.error("The 'results' list in the JSON is empty. Cannot determine defaults or process data.")
        sys.exit(1)

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
        vocab_size = None
        while vocab_size is None:  # Loop until we have a valid integer
            user_input = input("Please specify the vocabulary size of the model used (must be a positive integer): ")
            try:
                vocab_size_candidate = int(user_input)
                if vocab_size_candidate <= 0:
                    logging.warning("Vocabulary size must be positive. Please try again.")
                    # vocab_size remains None, loop continues
                else:
                    vocab_size = vocab_size_candidate  # Valid input received, store it
                    logging.info(f"Using provided vocabulary size: {vocab_size}")
            except ValueError:
                logging.warning(f"Invalid input '{user_input}'. Please enter an integer.")
                # vocab_size remains None, loop continues
        print("-" * 30)

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
            cumulative_entropy_rate, entropy_rate_per_token, token_relative_remaining_entropy = compute_entropy_rate(
                logprobs, number_logprobs, vocab_size
            )
            scored_answer = {
                "answer_text": answer_text,
                "cumulative_logprob": cumulative_logprob,
                "cumulative_entropy_rate": cumulative_entropy_rate,
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
