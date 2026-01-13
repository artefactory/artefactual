import argparse
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
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


def evaluate_with_pretrained_model(X, y, feature_cols, model_coeffs_file, plot_path=None):
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

    # CRITICAL: Use the same hyperparameters as the training model.
    model = LogisticRegression(penalty=None, max_iter=1000)

    # Perform a dummy fit to initialize model attributes correctly.
    # This sets the expected number of features and the .classes_ attribute.
    num_features = len(feature_cols)
    dummy_X = np.zeros((2, num_features))
    dummy_y = np.array([0, 1])
    model.fit(dummy_X, dummy_y)

    # Now, overwrite the learned parameters with the pre-trained ones.
    model.intercept_ = np.array([model_params["intercept"]])
    model.coef_ = np.array([[model_params["coefficients"][fn] for fn in feature_cols]])

    print("Evaluating model on the full dataset...")
    y_probs = model.predict_proba(X)[:, 1]

    pr_auc = average_precision_score(y, y_probs)
    roc_auc = roc_auc_score(y, y_probs)

    if plot_path:
        precision, recall, _ = precision_recall_curve(y, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker=".", label=f"PR Curve (AP = {pr_auc:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        try:
            plt.savefig(plot_path)
            print(f"\nPR curve plot saved to '{plot_path}'.")
        except OSError as e:
            print(f"Error saving plot: {e}")
        plt.close()

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
    parser.add_argument("-r", "--repetitions",
        type=int,
        default=1000,
        help="The number of bootstrap repetitions for training.",
    )
    parser.add_argument("-o", "--output_model_file",
        type=str,
        help="Path to save the output model parameters (JSON). Required for training.",
    )
    parser.add_argument("-m", "--model_coefficients",
        type=str,
        help="Optional path to a JSON file with pre-trained model coefficients for evaluation.",
    )
    parser.add_argument(
        "--plot_pr_curve",
        type=str,
        help="Optional path to save the PR curve plot image when evaluating a pre-trained model.",
    )
    args = parser.parse_args()

    try:
        df = pd.read_json(args.input_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading input file: {e}")
        return

    prefixes = ["mean", "max"]  # removed the min, that does not add much
    feature_cols = [f"{prefix}_rank_{i}" for prefix in prefixes for i in range(1, 16)]
    target_col = "judgment"

    if not all(col in df.columns for col in feature_cols) or target_col not in df.columns:
        print("Error: Input file is missing required feature or target columns.")
        return

    X = df[feature_cols]
    y = (~df[target_col]).astype(int)  # Invert judgment: 1 for hallucination detected (judged False)

    if args.model_coefficients:
        pr_auc, roc_auc = evaluate_with_pretrained_model(
            X, y, feature_cols, args.model_coefficients, args.plot_pr_curve
        )
        if pr_auc is not None:
            print("\n--- Pre-trained WEPR Model Evaluation Results ---")
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

        print("\n--- Bootstrap WEPR Evaluation Results ---")
        print(f"Mean PR-AUC: {mean_pr_auc:.4f}")
        print(f"Mean ROC-AUC: {mean_roc_auc:.4f}")
        print("------------------------------------")

        model_output = {
            "intercept": mean_intercept,
            "coefficients": dict(zip(feature_cols, mean_coefs, strict=False)),  # type: ignore
        }

        try:
            with open(args.output_model_file, "w", encoding="utf-8") as f:
                json.dump(model_output, f, indent=4)
            print(f"\nSuccessfully wrote model parameters to '{args.output_model_file}'.")
        except OSError as e:
            print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    main()
