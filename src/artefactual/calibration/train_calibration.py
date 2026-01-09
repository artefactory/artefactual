"""
Train a logistic regression model to calibrate uncertainty scores.

This script takes the output of rates_answers.py (a CSV with uncertainty scores and judgments),
trains a logistic regression model, and saves the coefficients in a JSON format suitable
for loading into the EPR scorer.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

MIN_CLASSES_FOR_TRAINING = 2


def train_calibration(input_file: str | Path, output_file: str | Path) -> None:
    """
    Train a logistic regression model to calibrate uncertainty scores.

    Args:
        input_file: Path to the CSV file containing 'uncertainty_score' and 'judgment'.
        output_file: Path to save the calibration weights (JSON).
    """
    # Load data
    df = pd.read_csv(input_file)

    if "uncertainty_score" not in df.columns or "judgment" not in df.columns:
        msg = "Input file must contain 'uncertainty_score' and 'judgment' columns."
        raise ValueError(msg)

    # Filter valid data
    # judgment can be True/False/None.
    df = df.dropna(subset=["judgment"])

    if df.empty:
        msg = "No valid data found (all judgments are None or file is empty)."
        raise ValueError(msg)

    # Convert judgment to target
    # We want to model P(hallucination), so target=1 if judgment is False (incorrect),
    # and target=0 if judgment is True (correct).
    def parse_judgment_to_target(val):
        s = str(val).lower()
        if s == "false":
            return 1
        if s == "true":
            return 0
        return None

    df["target"] = df["judgment"].apply(parse_judgment_to_target)

    # Drop any rows where parsing failed
    df = df.dropna(subset=["target"])
    # Cast to int to ensure numeric dtype for np.unique and model training
    df["target"] = df["target"].astype(int)
    x = df[["uncertainty_score"]].values
    y = df["target"].values

    if len(np.unique(y)) < MIN_CLASSES_FOR_TRAINING:
        msg = "Need both positive (False) and negative (True) judgments to train."
        raise ValueError(msg)

    logger.info(f"Training on {len(df)} samples.")

    # Train Logistic Regression
    clf = LogisticRegression(random_state=42)
    clf.fit(x, y)

    intercept = float(clf.intercept_[0])
    coef = float(clf.coef_[0][0])

    logger.info(f"Trained model: intercept={intercept}, coef={coef}")

    # Save weights
    # The EPR class expects 'mean_entropy' for the single coefficient.
    weights = {"intercept": intercept, "coefficients": {"mean_entropy": coef}}

    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(weights, f, indent=4)
    logger.info(f"Saved weights to {output_file}")
