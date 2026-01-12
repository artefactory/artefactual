import argparse
import json
import os


def update_main_answer(main_json_file, updates_jsonl_file, output_json_file):  # noqa: C901, PLR0911, PLR0912, PLR0915
    """
    Reads a main JSON file and a JSONL update file, and writes a new
    JSON file with the 'main_answer' field updated.
    """
    # --- Step 1: Read the JSONL file and create a lookup map ---
    # This map will store {query_id: generated_answer} for efficient lookups.
    updates_map = {}
    print(f"Reading updates from '{updates_jsonl_file}'...")

    if not os.path.exists(updates_jsonl_file):
        print(f"Error: The update file '{updates_jsonl_file}' was not found.")
        return

    try:
        with open(updates_jsonl_file, encoding="utf-8") as f_jsonl:
            for line in f_jsonl:
                # Ensure the line is not empty before parsing
                if line.strip():
                    update_record = json.loads(line)
                    query_id = update_record.get("query_id")
                    generated_answer = update_record.get("generated_answer")

                    # Ensure both required keys exist in the line
                    if query_id and generated_answer is not None:
                        updates_map[query_id] = generated_answer
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{updates_jsonl_file}': {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading '{updates_jsonl_file}': {e}")
        return

    print(f"Found {len(updates_map)} updates to apply.")

    # --- Step 2: Read the main JSON file ---
    print(f"\nReading main data from '{main_json_file}'...")

    if not os.path.exists(main_json_file):
        print(f"Error: The main JSON file '{main_json_file}' was not found.")
        return

    try:
        with open(main_json_file, encoding="utf-8") as f_json:
            main_data = json.load(f_json)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{main_json_file}': {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading '{main_json_file}': {e}")
        return

    # --- Step 3: Iterate and update the main data structure ---
    update_count = 0
    unmatched_count = 0

    # Check if 'results' key exists and is a list
    if "results" in main_data and isinstance(main_data["results"], list):
        for result in main_data["results"]:
            query_id = result.get("query_id")
            if query_id in updates_map:
                # Replace the main_answer with the one from the updates map
                result["main_answer"] = updates_map[query_id]
                update_count += 1
            else:
                unmatched_count += 1
    else:
        print("Error: The main JSON file does not have a 'results' list.")
        return

    # --- Step 4: Write the modified data to a new file ---
    print("\nWriting modified data...")
    try:
        with open(output_json_file, "w", encoding="utf-8") as f_out:
            json.dump(main_data, f_out, indent=4)
    except OSError as e:
        print(f"Error writing to file '{output_json_file}': {e}")
        return

    # --- Final Summary ---
    print("\n--- Processing Complete---")
    print(f"Successfully updated {update_count} records.")
    if unmatched_count > 0:
        print(f"{unmatched_count} records in '{main_json_file}' had no matching query_id in '{updates_jsonl_file}'.")
    print(f"The modified data has been saved to '{output_json_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update the 'main_answer' field in a JSON file based on a JSONL update file."
    )
    parser.add_argument("main_json_file", help="Path to the main JSON file.")
    parser.add_argument("updates_jsonl_file", help="Path to the JSONL update file.")
    parser.add_argument("output_json_file", help="Path to the output JSON file.")
    args = parser.parse_args()

    update_main_answer(args.main_json_file, args.updates_jsonl_file, args.output_json_file)
