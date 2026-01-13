import argparse
import json
import math
import pathlib


def calculate_features(full_info) -> dict | None:
    """
    Calculates the four features (mtp, avgtp, Mpd, mps) for a single generated answer,
    assuming the first token in each logprob list is the one that was sampled.

    Args:
        full_info (dict): A dictionary containing the 'answer_text' and 'detailed_logprobs'.

    Returns:
        dict: A dictionary containing the four calculated feature values, or None on error.
    """
    detailed_logprobs = full_info.get("detailed_logprobs", [])
    if not detailed_logprobs:
        return None

    # Lists to store probabilities for each token in the answer
    generated_token_probs = []
    deviations = []
    spreads_approx = []

    for candidates in detailed_logprobs:
        if not candidates:
            # print(f"Warning: Empty candidate list at token position {i}.")
            continue

        # The actually generated token (t_i) is always the first in the list.
        logprob_t_i = candidates[0].get("logprob")

        # Find the most likely token (v*), which has rank 1.
        v_star_info = next((cand for cand in candidates if cand.get("rank") == 1), None)

        if v_star_info is None:
            # This case should be rare, but we handle it for safety.
            # It could mean the rank 1 token is the same as the sampled token.
            if candidates[0].get("rank") == 1:
                v_star_info = candidates[0]
            else:
                # print(f"Warning: Could not find a token with rank 1 at position {i}. Skipping this token.")
                continue

        logprob_v_star = v_star_info.get("logprob")

        # Logprob of the least likely token (v-) from the provided list.
        # Used for the 'mps' approximation.
        logprob_v_minus_approx = candidates[-1].get("logprob")

        # Convert log probabilities to actual probabilities
        prob_t_i = math.exp(logprob_t_i)
        prob_v_star = math.exp(logprob_v_star)
        prob_v_minus_approx = math.exp(logprob_v_minus_approx)

        generated_token_probs.append(prob_t_i)

        # Calculate deviation for Mpd
        deviations.append(prob_v_star - prob_t_i)

        # Calculate approximate spread for mps
        spreads_approx.append(prob_v_star - prob_v_minus_approx)

    if not generated_token_probs:
        return None  # Return None if no tokens were successfully processed

    # --- Feature Calculation ---
    # 1. mtp: Minimum Token Probability
    mtp = min(generated_token_probs)

    # 2. avgtp: Average Token Probability
    avgtp = sum(generated_token_probs) / len(generated_token_probs)

    # 3. Mpd: Maximum Probability Deviation
    mpd = max(deviations)

    # 4. mps: Minimum Probability Spread (Approximated)
    # Note: Still an approximation as it doesn't use the full vocabulary.
    mps_approx = min(spreads_approx)

    return {
        "mtp": mtp,
        "avgtp": avgtp,
        "Mpd": mpd,
        "mps_approx": mps_approx
    }


def main() -> None:
    """
    Main function to read the input JSON, process it, and write the output JSON.
    """
    parser = argparse.ArgumentParser(description="Calculate probability-based features (HalluDetect) from generated answers.")
    parser.add_argument("input_file", help="Path to the input JSON file containing generation results with logprobs.")
    parser.add_argument("output_file", help="Path to save the output JSON file with calculated features.")
    args = parser.parse_args()

    input_filename = args.input_file
    output_filename = args.output_file

    print(f"Reading data from '{input_filename}'...")

    try:
        with pathlib.Path(input_filename).open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_filename}'. Please check the file format.")
        return

    results_list = data.get("results", [])
    output_data = []

    print(f"Found {len(results_list)} queries to process...")

    for result in results_list:
        query_id = result.get("query_id")

        if result.get("full_info"):
            full_info = result["full_info"][0]
            features = calculate_features(full_info)

            if features:
                output_entry = {"query_id": query_id, **features}
                output_data.append(output_entry)
            else:
                print(f"Skipping query_id '{query_id}' due to processing errors.")

    print(f"Writing {len(output_data)} results to '{output_filename}'...")

    try:
        with pathlib.Path(output_filename).open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        print("Done!")
    except OSError:
        print(f"Error: Could not write to output file '{output_filename}'.")


if __name__ == "__main__":
    main()
