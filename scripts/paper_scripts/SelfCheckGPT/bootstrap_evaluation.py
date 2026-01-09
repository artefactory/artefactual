import argparse
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# --- Configuration ---
# Set the number of bootstrap repetitions
N_REPETITIONS = 1000

SEED = 42

def true_bootstrap_evaluation(input_json_file):
    """
    Performs a true bootstrap evaluation by creating train sets with replacement
    and using the out-of-bag (OOB) samples for testing.
    """
    # --- Step 1: Load and Prepare Data ---
    print(f"Loading data from '{input_json_file}'...")

    if not os.path.exists(input_json_file):
        return f"Error: The file '{input_json_file}' was not found."

    try:
        with open(input_json_file, encoding="utf-8") as f:
            data = json.load(f)

        scores = [item["selfcheck_bertscore"] for item in data]
        judgments = [item["judgment"] for item in data]

        X = np.array(scores).reshape(-1, 1)
        y = np.array(judgments).astype(str)  # type: ignore # Invert judgment: 1 for hallucination detected (judged False)
        y = np.array([0 if j == "true" else 1 for j in y])  # Now 1 indicates hallucination detected

        print("First 10 scores and judgments for verification:")
        print(judgments[0:10], "corresponding int:", y[0:10])
    except (Exception, KeyError) as e:
        return f"An error occurred during data loading or parsing for file {input_json_file}: {e}"

    print(f"Data loaded successfully. Found {len(data)} records.")

    # --- Step 2: Run the bootstrap loop ---
    roc_auc_scores = []
    pr_auc_scores = []
    n_samples = len(y)
    all_indices = np.arange(n_samples)

    print(f"\nRunning {N_REPETITIONS} true bootstrap repetitions...")

    np.random.seed(SEED)

    for _ in tqdm(range(N_REPETITIONS), desc="Evaluating Bootstrap Samples", leave=False):
        # a. Create the bootstrap sample (training set) by sampling WITH replacement
        train_indices = np.random.choice(all_indices, size=n_samples, replace=True)
        X_train, y_train = X[train_indices], y[train_indices]

        # b. Identify the out-of-bag (OOB) sample (test set)
        oob_indices = np.setdiff1d(all_indices, np.unique(train_indices))

        # Ensure the OOB set has samples and contains more than one class for evaluation
        if len(oob_indices) == 0 or len(np.unique(y[oob_indices])) < 2:
            continue  # Skip this iteration as AUC is not defined

        X_test, y_test = X[oob_indices], y[oob_indices]

        # c. Train and evaluate the model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc_scores.append(roc_auc_score(y_test, y_pred_proba))
        pr_auc_scores.append(average_precision_score(y_test, y_pred_proba))

    print("Bootstrap evaluation complete.")

    # --- Step 3: Calculate and display the results ---
    mean_roc_auc = np.mean(roc_auc_scores)
    std_roc_auc = np.std(roc_auc_scores)

    mean_pr_auc = np.mean(pr_auc_scores)
    std_pr_auc = np.std(pr_auc_scores)

    ci_roc_auc = (mean_roc_auc - 1.96 * std_roc_auc, mean_roc_auc + 1.96 * std_roc_auc)
    ci_pr_auc = (mean_pr_auc - 1.96 * std_pr_auc, mean_pr_auc + 1.96 * std_pr_auc)

    result_str = f"--- Results for {os.path.basename(input_json_file)} ---\n"
    result_str += f"Based on {len(roc_auc_scores)} valid out-of-bag evaluations:\n\n"
    result_str += "ROC-AUC Score:\n"
    result_str += f"  - Mean:               {mean_roc_auc:.4f}\n"
    result_str += f"  - Standard Deviation: {std_roc_auc:.4f}\n"
    result_str += f"  - 95% CI:             ({ci_roc_auc[0]:.4f}, {ci_roc_auc[1]:.4f})\n\n"
    result_str += "PR-AUC Score (Average Precision):\n"
    result_str += f"  - Mean:               {mean_pr_auc:.4f}\n"
    result_str += f"  - Standard Deviation: {std_pr_auc:.4f}\n"
    result_str += f"  - 95% CI:             ({ci_pr_auc[0]:.4f}, {ci_pr_auc[1]:.4f})\n"
    result_str += "------------------------------------------------------\n"

    return result_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform bootstrap evaluation on all '*judged.json' files in a directory."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing the '*judged.json' files.",
        default="rated_data/",
        nargs="?",
    )
    parser.add_argument(
        "output_file",
        help="Path to the output text file for results.",
        default="SCGPT_classif_results.txt",
        nargs="?",
    )
    args = parser.parse_args()

    search_path = os.path.join(args.input_dir, "*judged.json")
    files_to_process = glob.glob(search_path)

    if not files_to_process:
        print(f"No '*judged.json' files found in '{args.input_dir}'.")
        exit()

    all_results = []
    for file_path in tqdm(files_to_process, desc="Processing files"):
        result = true_bootstrap_evaluation(file_path)
        all_results.append(result)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(all_results))

    print(f"\nAll results have been written to '{args.output_file}'.")
