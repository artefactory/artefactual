import argparse
import json
import math


def calculate_entropic_features(input_file_path, output_file_path):
    """
    Reads LLM output, calculates mean, max, and min entropic contributions
    for each of the top 15 token ranks, and saves the results as features.

    Args:
        input_file_path (str): The path to the input JSON file.
        output_file_path (str): The path where the output JSON will be saved.
    """
    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f'Error: Input file not found at "{input_file_path}"')
        return
    except json.JSONDecodeError:
        print(f'Error: Could not decode JSON from "{input_file_path}"')
        return

    final_results = []

    for result in data.get("results", []):
        query_id = result.get("query_id")
        if not query_id:
            continue

        # This dictionary will store a list of contributions for each rank
        rank_contributions = {i: [] for i in range(1, 16)}

        for answer_info in result.get("full_info", []):
            for token_step in answer_info.get("detailed_logprobs", []):
                for token_data in token_step:
                    rank = token_data.get("rank")
                    logprob = token_data.get("logprob")

                    if rank is not None and logprob is not None and 1 <= rank <= 15:
                        p = math.exp(logprob)
                        contribution = -p * logprob
                        rank_contributions[rank].append(contribution)

        # Create a dictionary for the flattened output
        query_output = {"query_id": query_id}
        list_maximum = []
        list_minimum = []
        # Calculate mean, max, and min for each rank and add them to the output
        for rank, contributions in rank_contributions.items():
            if contributions:
                mean_val = sum(contributions) / len(contributions)
                max_val = max(contributions)
                list_maximum.append(max_val)
                min_val = min(contributions)
                list_minimum.append(min_val)
            else:
                # If a rank never appeared, its stats are 0
                mean_val = 0.0
                max_val = 0.0
                min_val = 0.0

            # Add the new features to the output dictionary with descriptive names
            query_output[f"mean_rank_{rank}"] = mean_val
            query_output[f"max_rank_{rank}"] = max_val
            query_output[f"min_rank_{rank}"] = min_val
        # query_output["max"] = max(list_maximum)
        # query_output["min"] = min(list_minimum)

        final_results.append(query_output)

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"Successfully processed {len(final_results)} queries.")
    print(f"Output saved to \"{output_file_path}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate mean, max, and min entropic contributions from LLM logprobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSON file containing LLM generation data."
    )

    parser.add_argument(
        "output_file",
        type=str,
        help="Path for the output JSON file where features will be saved."
    )

    args = parser.parse_args()
    calculate_entropic_features(args.input_file, args.output_file)
