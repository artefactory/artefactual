#!/usr/bin/env python3
"""
Script to train calibration.
Moves execution logic from src/artefactual/calibration/train_calibration.py
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

from artefactual.calibration.train_calibration import train_calibration

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Train calibration weights.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV from rates_answers.py")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSON weights")
    args = parser.parse_args()

    try:
        train_calibration(args.input_file, args.output_file)
    except ValueError:
        logger.exception("Calibration failed")
        sys.exit(1)
