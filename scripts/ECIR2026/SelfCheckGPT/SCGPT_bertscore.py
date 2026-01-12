# score_fast.py
import argparse
import json
import logging
import os
import time

import bert_score
import torch

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("transformers").setLevel(logging.ERROR)


def main(args):
    """
    Main function to load generated text, score it in a batched fashion, and save the results.
    """
    # --- Load Generated Data ---
    logging.info(f"Loading generated data from {args.input_file}")
    try:
        with open(args.input_file, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        return

    results = data.get("results", [])
    if not results:
        logging.warning("No results found in the input file.")
        return

    # --- 1. Prepare Data for Batching ---
    logging.info("Preparing data for batch processing...")
    all_candidates = []
    all_references = []
    samples_per_query = []

    for entry in results:
        main_answer = entry.get("main_answer", "").strip()
        sampled_answers = entry.get("sampled_answers", [])

        # Ensure we have a non-empty reference
        ref = main_answer if main_answer else "[PAD]"

        if sampled_answers:
            # Add all samples from this query to the main list of candidates
            all_candidates.extend([s.strip() if s.strip() else "[PAD]" for s in sampled_answers])
            # Add the main answer, repeated for each sample, to the references list
            all_references.extend([ref] * len(sampled_answers))
            # Keep track of how many samples this query had
            samples_per_query.append(len(sampled_answers))
        else:
            # If no samples, there's nothing to score
            samples_per_query.append(0)

    if not all_candidates:
        logging.error("No sampled answers found across all queries. Cannot perform scoring.")
        # Still, we should write the output file with null scores.
        for entry in results:
            entry["selfcheck_bertscore"] = None
    else:
        # --- 2. Perform Single Batched Scoring Call ---
        logging.info(f"Scoring {len(all_candidates)} samples in a single batch...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        start_time = time.time()
        P, R, F1_scores = bert_score.score(
            all_candidates,
            all_references,
            lang="en",
            verbose=True,  # Set to True to see the progress bar from bert_score
            rescale_with_baseline=True,
            device=device,
        )
        end_time = time.time()
        logging.info(f"Batch scoring completed in {end_time - start_time:.2f} seconds.")

        # --- 3. Assign Scores Back to Queries ---
        logging.info("Assigning scores back to original queries...")
        current_score_idx = 0
        for i, entry in enumerate(results):
            num_samples = samples_per_query[i]
            if num_samples > 0:
                # Get the slice of scores corresponding to this query's samples
                query_scores = F1_scores[current_score_idx : current_score_idx + num_samples]
                # Calculate the average score for the query
                avg_f1_score = query_scores.mean().item()  # type: ignore
                # The final score is 1 - F1
                entry["selfcheck_bertscore"] = 1.0 - avg_f1_score
                current_score_idx += num_samples
            else:
                # No samples to compare against
                entry["selfcheck_bertscore"] = None

    # --- Save Results ---
    base_name = os.path.splitext(args.input_file)[0]
    output_file = f"{base_name}_Bertscored.json"

    logging.info(f"Writing scored results to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logging.info("Scoring complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficiently score LLM responses using batched SelfCheckGPT-BERTScore.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file from the generation script.")
    args = parser.parse_args()
    main(args)
