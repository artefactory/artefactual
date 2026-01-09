import argparse
import json
import os


def create_combined_file(source_json_file, source_jsonl_file, output_file):
    """
    Combines 'query_id', 'selfcheck_bertscore', and 'judgment' from two
    different source files into a new, single JSON file.
    """
    # --- Step 1: Create a lookup map for 'judgment' from the JSONL file ---
    # This map will store {query_id: judgment} for fast lookups.
    judgment_map = {}
    print(f"Reading judgments from '{source_jsonl_file}'...")

    if not os.path.exists(source_jsonl_file):
        print(f"Error: The file '{source_jsonl_file}' was not found.")
        return

    try:
        with open(source_jsonl_file, encoding="utf-8") as f_jsonl:
            for line in f_jsonl:
                if line.strip():  # Avoid processing empty lines
                    record = json.loads(line)
                    query_id = record.get("query_id")
                    judgment = record.get("judgment")
                    # We need both keys. 'judgment' can be `False`, so we check for `None`.
                    if query_id and judgment is not None:
                        judgment_map[query_id] = judgment
    except Exception as e:
        print(f"An error occurred while reading '{source_jsonl_file}': {e}")
        return

    print(f"Found {len(judgment_map)} judgments to combine.")

    # --- Step 2: Read the source JSON file to get bertscore ---
    print(f"\nReading scores from '{source_json_file}'...")

    if not os.path.exists(source_json_file):
        print(f"Error: The file '{source_json_file}' was not found.")
        return

    try:
        with open(source_json_file, encoding="utf-8") as f_json:
            source_data = json.load(f_json)
    except Exception as e:
        print(f"An error occurred while reading '{source_json_file}': {e}")
        return

    # --- Step 3: Iterate, combine data, and build the new list ---
    combined_results = []
    records_processed = 0
    records_skipped = 0

    if "results" in source_data and isinstance(source_data.get("results"), list):
        for item in source_data["results"]:
            query_id = item.get("query_id")
            bertscore = item.get("selfcheck_bertscore")

            # Check if the item has all required data and a match in our judgment map
            if query_id and bertscore is not None and query_id in judgment_map:
                new_record = {
                    "query_id": query_id,
                    "selfcheck_bertscore": bertscore,
                    "judgment": judgment_map[query_id]
                }
                combined_results.append(new_record)
                records_processed += 1
            else:
                records_skipped += 1
    else:
        print(f"Error: The key 'results' was not found or is not a list in '{source_json_file}'.")
        return

    # --- Step 4: Write the new list to the output JSON file ---
    try:
        with open(output_file, "w", encoding="utf-8") as f_out:
            json.dump(combined_results, f_out, indent=4)
    except OSError as e:
        print(f"Error writing to file '{output_file}': {e}")
        return

    # --- Final Summary ---
    print("\n--- Processing Complete! ---")
    print(f"Successfully created {records_processed} combined records.")
    if records_skipped > 0:
        print(f"Skipped {records_skipped} records due to missing data or unmatched query_ids.")
    print(f"The new data has been saved to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combines 'query_id', 'selfcheck_bertscore', and 'judgment' from two source files into a new, single JSON file."
    )
    parser.add_argument("source_json_file", help="Path to the source JSON file (contains 'selfcheck_bertscore').")
    parser.add_argument("source_jsonl_file", help="Path to the source JSONL file (contains 'judgment').")
    parser.add_argument("output_file", help="Path to the output JSON file.")
    args = parser.parse_args()

    create_combined_file(args.source_json_file, args.source_jsonl_file, args.output_file)
