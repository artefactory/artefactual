import argparse
import json
import sys


def merge_data(features_filepath, annotations_filepath, output_filepath):
    """
    Merges features and annotations based on query_id.

    Args:
        features_filepath (str): Path to the JSON file with calculated features.
        annotations_filepath (str): Path to the JSON-Line file with judgment annotations.
        output_filepath (str): Path for the output merged JSON file.
    """
    # --- Step 1: Load the features data into a dictionary for fast lookup ---
    print(f"Reading features from '{features_filepath}'...")
    try:
        with open(features_filepath, encoding="utf-8") as f:
            features_data = json.load(f)

        # Create a map: {query_id: {feature_data}}
        features_map = {item["query_id"]: item for item in features_data}
        print(f"Loaded {len(features_map)} feature entries.")

    except FileNotFoundError:
        print(f"Error: Features file not found at '{features_filepath}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{features_filepath}'.")
        return

    # --- Step 2: Read the annotations file and merge with features ---
    print(f"Reading annotations from '{annotations_filepath}'...")
    merged_data = []
    found_matches = 0

    try:
        with open(annotations_filepath, encoding="utf-8") as f:
            for line in f:
                try:
                    annotation = json.loads(line)
                    query_id = annotation.get("query_id")
                    judgment = annotation.get("judgment")

                    # Check if we have features for this query_id
                    if query_id in features_map:
                        # Get the corresponding features
                        features = features_map[query_id]

                        # Create the new merged entry
                        merged_entry = {
                            **features,  # Unpacks the features dict
                            "judgment": judgment
                        }
                        # Ensure query_id is present and correct
                        merged_entry["query_id"] = query_id

                        merged_data.append(merged_entry)
                        found_matches += 1

                except json.JSONDecodeError:
                    print("Warning: Skipping a line in annotations file that is not valid JSON.")
                    continue

    except FileNotFoundError:
        print(f"Error: Annotations file not found at '{annotations_filepath}'")
        return

    print(f"Found {found_matches} matching entries between the two files.")

    # --- Step 3: Write the merged data to the output file ---
    print(f"Writing merged data to '{output_filepath}'...")
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=4)
        print("Done!")
    except OSError:
        print(f"Error: Could not write to output file '{output_filepath}'.")


def main():
    """
    Main function to handle command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Merge calculated features with judgment annotations.")
    parser.add_argument("features_file", help="Path to the JSON file with calculated features.")
    parser.add_argument("annotations_file", help="Path to the JSONL file with judgment annotations.")
    parser.add_argument("output_file", help="Path for the output merged JSON file.")
    args = parser.parse_args()

    merge_data(args.features_file, args.annotations_file, args.output_file)


if __name__ == "__main__":
    main()
