import argparse
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample
from tqdm import tqdm

SEED = 42


def load_data(filepath):
    """
    Loads the merged training data from a JSON file.

    Args:
        filepath (str): The path to the training_data.json file.

    Returns:
        tuple: A tuple containing:
               - X (np.ndarray): The feature matrix.
               - y (np.ndarray): The target vector (0s and 1s).
               - feature_names (list): The names of the feature columns.
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'.")
        sys.exit(1)

    if not data:
        print("Error: The JSON file is empty.")
        sys.exit(1)

    # Define the order of features to ensure consistency
    feature_names = ["mtp", "avgtp", "Mpd", "mps_approx"]

    # Extract features (X) and target (y)
    X = np.array([[item[fn] for fn in feature_names] for item in data])
    y = np.array([0 if item["judgment"] else 1 for item in data])  # 1 indicates hallucination

    return X, y, feature_names


def evaluate_with_bootstrap(X, y, n_iterations=1000):
    """
    Evaluates a Logistic Regression model using bootstrap cross-validation.

    For each iteration, it trains on a bootstrap sample, tests on the
    out-of-bag (OOB) samples, and stores the model parameters.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        n_iterations (int): The number of bootstrap repetitions.

    Returns:
        tuple: A tuple containing the mean PR-AUC, mean ROC-AUC,
               mean coefficients, and mean intercept.
    """
    n_samples = X.shape[0]
    pr_auc_scores = []
    roc_auc_scores = []
    model_coefs = []
    model_intercepts = []

    print(f"Starting bootstrap evaluation with {n_iterations} iterations...")
    np.random.seed(SEED)
    for _ in tqdm(range(n_iterations), desc="Bootstrap Iterations"):
        # Create bootstrap sample (train set) and out-of-bag sample (test set)
        train_indices = resample(np.arange(n_samples))
        oob_indices = np.array([idx for idx in np.arange(n_samples) if idx not in train_indices])

        # Ensure the OOB set is usable
        if len(oob_indices) == 0 or len(np.unique(y[oob_indices])) < 2:
            # Skip iteration if OOB is empty or has only one class
            continue

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[oob_indices], y[oob_indices]

        # Initialize and train the model
        model = LogisticRegression(penalty=None, max_iter=1000)
        model.fit(X_train, y_train)

        # Predict probabilities for the positive class on the OOB set
        y_probs = model.predict_proba(X_test)[:, 1]

        # Calculate and store scores
        pr_auc_scores.append(average_precision_score(y_test, y_probs))
        roc_auc_scores.append(roc_auc_score(y_test, y_probs))

        model_coefs.append(model.coef_[0])  # model.coef_ is [[c1, c2, ...]]
        model_intercepts.append(model.intercept_[0])  # model.intercept_ is [i]

    # Calculate the mean of the scores and parameters
    mean_pr_auc = np.mean(pr_auc_scores)
    mean_roc_auc = np.mean(roc_auc_scores)
    mean_coefs = np.mean(model_coefs, axis=0)
    mean_intercept = np.mean(model_intercepts)

    return mean_pr_auc, mean_roc_auc, mean_coefs, mean_intercept


def evaluate_with_pretrained_model(X, y, feature_names, model_coeffs_file):
    """
    Evaluates a pre-trained Logistic Regression model on a new dataset.

    Args:
        X (np.ndarray): The feature matrix of the new dataset.
        y (np.ndarray): The target vector of the new dataset.
        feature_names (list): The names of the features, in order.
        model_coeffs_file (str): Path to the JSON file with model coefficients.

    Returns:
        tuple: A tuple containing the PR-AUC and ROC-AUC scores.
    """
    print(f"Loading pre-trained model parameters from '{model_coeffs_file}'...")
    try:
        with open(model_coeffs_file, encoding="utf-8") as f:
            model_params = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading model coefficients file: {e}")
        sys.exit(1)

    # Create a new model and set its parameters
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.intercept_ = np.array([model_params["intercept"]])
    # Ensure coefficients are in the correct order
    model.coef_ = np.array([[model_params["coefficients"][fn] for fn in feature_names]])
    model.classes_ = np.array([0, 1])  # Manually set classes attribute

    print("Evaluating model on the full dataset...")
    y_probs = model.predict_proba(X)[:, 1]

    pr_auc = average_precision_score(y, y_probs)
    roc_auc = roc_auc_score(y, y_probs)

    return pr_auc, roc_auc


def main():
    """
    Main function to run the evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a logistic regression model on feature data, with an option to use a pre-trained model."
    )
    parser.add_argument("training_data_file", help="Path to the training_data.json file.")
    parser.add_argument(
        "-o", "--output_model_file", help="Path to save the output model parameters (JSON)."
    )
    parser.add_argument(
        "-m", "--model_coefficients",
        help="Optional path to a JSON file with pre-trained model coefficients to evaluate on the dataset.",
    )
    args = parser.parse_args()

    X, y, feature_names = load_data(args.training_data_file)
    print("Data loaded successfully.")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Features used: {', '.join(feature_names)}\n")

    if args.model_coefficients:
        # --- Evaluation with a pre-trained model ---
        pr_auc, roc_auc = evaluate_with_pretrained_model(
            X, y, feature_names, args.model_coefficients
        )
        print("\n--- Pre-trained Model Evaluation Results ---")
        print(f"PR-AUC on full dataset:  {pr_auc:.4f}")
        print(f"ROC-AUC on full dataset: {roc_auc:.4f}")
        print("------------------------------------------")

    else:
        # --- Original bootstrap evaluation ---
        if not args.output_model_file:
            print("Error: --output_model_file is required when not using --model_coefficients.")
            sys.exit(1)

        mean_pr_auc, mean_roc_auc, mean_coefs, mean_intercept = evaluate_with_bootstrap(
            X, y, n_iterations=1000
        )

        print("\n--- Bootstrap Evaluation Results ---")
        print(f"Mean ROC-AUC:     {mean_roc_auc:.4f}")
        print(f"Mean PR-AUC:      {mean_pr_auc:.4f}")
        print("------------------------------------\n")

        model_output = {
            "intercept": mean_intercept,
            "coefficients": dict(zip(feature_names, mean_coefs)),
        }

        print(f"Writing mean model parameters to '{args.output_model_file}'...")
        try:
            with open(args.output_model_file, "w", encoding="utf-8") as f:
                json.dump(model_output, f, indent=4)
            print("Successfully wrote model parameters.")
        except IOError as e:
            print(f"Error: Could not write to file '{args.output_model_file}'.\n{e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
