import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

# Suppress convergence warnings from scikit-learn for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42


def train_and_evaluate_with_bootstrap(X, y, n_repetitions=1000):
    """
    Trains and evaluates a logistic regression model using bootstrap validation.

    This function performs bootstrap sampling, computes the mean PR-AUC and
    ROC-AUC scores on out-of-bag samples, and returns the mean model parameters.

    Args:
        X (pd.DataFrame): The feature DataFrame.
        y (pd.Series): The target Series.
        feature_cols (list): The names of the feature columns.
        n_repetitions (int): The number of bootstrap repetitions.

    Returns:
        tuple: A tuple containing mean PR-AUC, ROC-AUC, coefficients, and intercept.
    """
    n_samples = len(X)
    all_indices = np.arange(n_samples)

    roc_auc_scores = []
    pr_auc_scores = []
    model_coefs = []
    model_intercepts = []

    print(f"Starting bootstrap evaluation with {n_repetitions} repetitions...")
    np.random.seed(SEED)
    for _ in tqdm(range(n_repetitions), desc="Bootstrap Iterations"):
        train_indices = np.random.choice(all_indices, size=n_samples, replace=True)
        test_indices = np.setdiff1d(all_indices, np.unique(train_indices))

        if len(test_indices) < 2 or len(np.unique(y.iloc[test_indices])) < 2:
            continue

        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]


        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        pr_auc_scores.append(average_precision_score(y_test, y_pred_proba))
        model_coefs.append(model.coef_[0])
        model_intercepts.append(model.intercept_[0])

    if not roc_auc_scores:
        return None, None, None, None

    mean_roc_auc = np.mean(roc_auc_scores)
    mean_pr_auc = np.mean(pr_auc_scores)
    mean_coefs = np.mean(model_coefs, axis=0)
    mean_intercept = np.mean(model_intercepts)

    return mean_roc_auc, mean_pr_auc, mean_coefs, mean_intercept


def evaluate_with_pretrained_model(X, y, feature_cols, model_coeffs_file):
    """
    Evaluates a pre-trained Logistic Regression model on a new dataset.
    """
    print(f"Loading pre-trained model parameters from '{model_coeffs_file}'...")
    try:
        with open(model_coeffs_file, encoding="utf-8") as f:
            model_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading model coefficients file: {e}")
        return None, None

    model = LogisticRegression(penalty=None, max_iter=1000)
    model.intercept_ = np.array([model_params["intercept"]])
    model.coef_ = np.array([[model_params["coefficients"][fn] for fn in feature_cols]])
    model.classes_ = np.array([0, 1])

    print("Evaluating model on the full dataset...")
    y_probs = model.predict_proba(X)[:, 1]

    pr_auc = average_precision_score(y, y_probs)
    roc_auc = roc_auc_score(y, y_probs)

    return pr_auc, roc_auc


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a logistic regression model with bootstrap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the JSON file containing features and judgments.",
    )
    parser.add_argument(
        "-r", "--repetitions",
        type=int,
        default=1000,
        help="The number of bootstrap repetitions for training.",
    )
    parser.add_argument(
        "-o", "--output_model_file",
        type=str,
        help="Path to save the output model parameters (JSON). Required for training.",
    )
    parser.add_argument(
        "-m", "--model_coefficients",
        type=str,
        help="Optional path to a JSON file with pre-trained model coefficients for evaluation.",
    )
    args = parser.parse_args()

    try:
        df = pd.read_json(args.input_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading input file: {e}")
        return

    prefixes = ["mean"]  # removed the max and min for EPR
    feature_cols = [f"{prefix}_rank_{i}" for prefix in prefixes for i in range(1, 16)]
    target_col = "judgment"

    if not all(col in df.columns for col in feature_cols) or target_col not in df.columns:
        print("Error: Input file is missing required feature or target columns.")
        return

    x_full = df[feature_cols]
    X = x_full.mean(axis=1).to_frame(name="mean_entropy")
    y = (~df[target_col]).astype(int)  # Invert judgment: 1 for hallucination detected (judged False)

    if args.model_coefficients:
        pr_auc, roc_auc = evaluate_with_pretrained_model(
            X, y, ["mean_entropy"], args.model_coefficients
        )
        if pr_auc is not None:
            print("\n--- Pre-trained EPR Model Evaluation Results ---")
            print(f"Mean PR-AUC on full dataset:  {pr_auc:.4f}")
            print(f"Mean ROC-AUC on full dataset: {roc_auc:.4f}")
            print("------------------------------------------")
    else:
        if not args.output_model_file:
            print("Error: --output_model_file is required for training.")
            return

        mean_roc_auc, mean_pr_auc, mean_coefs, mean_intercept = train_and_evaluate_with_bootstrap(
            X, y, args.repetitions
        )

        if mean_roc_auc is None:
            print("Error: Could not compute scores. The dataset might be too small.")
            return

        print("\n--- Bootstrap EPR Evaluation Results ---")
        print(f"Mean PR-AUC: {mean_pr_auc:.4f}")
        print(f"Mean ROC-AUC: {mean_roc_auc:.4f}")
        print("------------------------------------")

        model_output = {
            "intercept": mean_intercept,
            "coefficients": dict(zip(["mean_entropy"], mean_coefs, strict=False)),  # type: ignore
        }

        try:
            with open(args.output_model_file, "w", encoding="utf-8") as f:
                json.dump(model_output, f, indent=4)
            print(f"\nSuccessfully wrote model parameters to '{args.output_model_file}'.")
        except OSError as e:
            print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    main()
