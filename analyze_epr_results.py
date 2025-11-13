#!/usr/bin/env python3
"""
Analyze EPR results from demo_uncertainty.py output.

This script reads the JSON output and creates a summary showing EPR scores
in descending order for each question.

Usage:
    python analyze_epr_results.py <input_json_file> [--output OUTPUT_FILE]

Example:
    python analyze_epr_results.py generation_test/DEMO_Ministral-8B-Instruct-2410_1.0_10_with_EPR.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


def load_results(input_file: str) -> list[dict[str, Any]]:
    """Loads the JSON results from the file."""
    with open(input_file, encoding="utf-8") as f:
        return json.load(f)


def process_epr_scores(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extracts and sorts questions based on their EPR scores.
    Returns a list of dictionaries containing question info and sorted scores.
    """
    processed_results = []
    for entry in data:
        # Extract relevant fields (assuming structure based on context)
        question = entry.get("question", "Unknown Question")
        # You might need to adjust these keys based on your actual JSON structure
        epr_score = entry.get("epr_score", 0.0)
        processed_results.append({
            "question": question,
            "epr_score": epr_score,
            "raw_entry": entry,  # Keep reference if needed
        })

    # Sort by EPR score descending
    return sorted(processed_results, key=lambda x: x["epr_score"], reverse=True)


def write_summary(results: list[dict[str, Any]], output_file: Path):
    """Formats and writes the summary to the output file."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("EPR Analysis Summary\n")
        f.write(f"Total Questions: {len(results)}\n")
        f.write("-" * 40 + "\n\n")

        for i, item in enumerate(results, 1):
            f.write(f"{i}. Score: {item['epr_score']:.4f}\n")
            f.write(f"   Question: {item['question']}\n")
            f.write("\n")


def analyze_epr_results(input_file: str, output_file: str | None = None):
    """
    Analyze EPR results and create a summary file.

    Args:
        input_file: Path to the JSON output from demo_uncertainty.py
        output_file: Path to save the summary (optional, auto-generated if None)
    """
    # Determine output path
    input_path = Path(input_file)
    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_EPR_SUMMARY.txt"
    else:
        output_path = Path(output_file)

    # Orchestrate the workflow
    try:
        data = load_results(input_file)
        sorted_results = process_epr_scores(data)
        write_summary(sorted_results, output_path)
    except Exception as e:
        logging.error(f"Failed to analyze results: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze EPR results from demo_uncertainty.py output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", type=str, help="Path to the JSON output file from demo_uncertainty.py")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save the summary file (default: auto-generated)"
    )

    args = parser.parse_args()

    # Check input file exists
    if not Path(args.input_file).exists():
        logging.error(f"Input file not found: {args.input_file}")
        return 1

    analyze_epr_results(args.input_file, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
