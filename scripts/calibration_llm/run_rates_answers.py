#!/usr/bin/env python3
"""
Script to rate answers.
Moves execution logic from src/artefactual/calibration/rates_answers.py
"""
import argparse
import logging
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from artefactual.calibration.rates_answers import RatingConfig, rate_answers

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Rate answers using a judge LLM.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="mistralai/Ministral-8B-Instruct-2410",
        help="Path to the judge model.",
    )
    parser.add_argument("--output_file", type=str, help="Path to save the output DataFrame (CSV).")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if input_path.exists():
        config = RatingConfig(input_file=args.input_file, judge_model_path=args.model_path)
        df = rate_answers(config)
        logger.info("Rated answers head:\n%s", df.head(10))
        logger.info(f"Total rated: {len(df)}")
        if args.output_file:
            df.to_csv(args.output_file)
            logger.info(f"Saved results to {args.output_file}")
    else:
        logger.error(f"Input file {args.input_file} not found.")
        sys.exit(1)
