#!/usr/bin/env python3
"""
Script to generate entropy dataset.
Moves execution logic from src/artefactual/calibration/outputs_entropy.py
"""
import sys
from pathlib import Path

# Add src to path to ensure we can import artefactual if not installed in env
# Assuming script is in scripts/calibration_llm/ and src is ../../src
project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from artefactual.calibration.outputs_entropy import GenerationConfig, generate_entropy_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate entropy dataset.")
    parser.add_argument("--input_file", type=str, default="sample_qa_data.json", help="Path to the input JSON file relative to project root.")
    parser.add_argument("--output_path", type=str, default="outputs/", help="Path to save the output dataset.")
    args = parser.parse_args()
    config = GenerationConfig()
    # Using absolute path as it was in the original file, or relative to project root
    input_path = project_root / args.input_file
    generate_entropy_dataset(
        input_path=input_path,
        output_path=args.output_path,
        config=config,
    )
